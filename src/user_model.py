from typing import Iterable, Dict, Callable, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import random


class UserModel:
    
    def __init__(self, ground_truth: Tensor, brush_sizes=torch.arange(2,7)):
        super().__init__()
        
        # globals
        self.gt = ground_truth.float() # nd array or float tensor?
        self.brush_sizes = brush_sizes
        
        
        # statistics to track
        self.annotated_pixels = None
        self.annotated_slices = None
        
    
    def _sum_l1_per_slice(self, volume: Tensor) -> Tensor:
        """ Find sum over all slices in each direction

        Parameters
        ----------
        volume : Tensor 
            shape L x W x H with L = W = H  

        Returns
        -------
        out : Tensor
            (N, n) shaped array holding slice sums, where N is
            the dimensionality (3) and n the number of slices in each
            direction (L, W, H). Expects zero padding to ensure same length
            across dimensions.
        """

        assert((volume.shape[0] == volume.shape[1]) and (volume.shape[0] == volume.shape[2]))

        # dimensionality and array of possible axis for volume
        dims    = len(volume.shape)
        indices = torch.arange(dims)
        sums    = torch.zeros((dims, volume.shape[0]))

        # for each direction, sum values in each slice
        for dim in range(dims):
            axis       = tuple(indices[indices != dim])
            slice_sums = volume.sum(axis=axis)
            sums[dim]  = slice_sums
        return sums
        
        

    def _order_slices_by_sum(self, slice_sums: Tensor) -> Union[np.array, np.array]:
        """ Order slices by their overall sum
        Note: numpy dependent, because np.unravel_index exists

        Parameters
        ----------
        slice_sums : Tensor
            shape (N, n) with N=3 and n=L=H=W

        Returns
        -------
        axis : 1d array
            axis of slices in descending order w.r.t.
            their slice sum
        slices : 1d array
            slice index of slices in descending order
            w.r.t. their slice sum
        """

        length = slice_sums.shape[0]

        # calculate direction (axis) and indices of slices in 
        # descending order w.r.t. their slice sum
        sorted_slice_indices = torch.argsort(slice_sums.flatten(), descending=True)
        # Note: Unravel_index is not yet implemented in torch, but a requested feature
        #       as of Dec 2020. Maybe add later. 
        axis, slices = np.unravel_index(sorted_slice_indices, slice_sums.shape)
        
        return axis, slices


    def _slice_samples_per_class(self, slc: Tensor, inverse_frequencies: Tensor,
                                 n: int) -> Tensor:
        """ samples seeds for each class in a slice from the
            error map, weighted by inverse class frequencies

        Parameters
        ----------
        slc : Tensor
            slice from error map, shape n_classes x W x H

        inverse_frequencies : Tensor
            inverse class frequencies from ground truth, shape n_classes x 1 x 1 x 1

        n : int
            number of seeds

        Returns
        -------
        n_samples : Tensor
            shape n_classes

        """
        # omit sign for number of misclassifications
        slc_abs = torch.abs(slc)

        # calculate proportions by dividing total number of 
        # misclassifications by class frequency for each class
        # and then normalize to stochastic vector
        total                  = slc_abs.sum(dim=(1,2))
        proportions            = total * inverse_frequencies.flatten()
        proportions_normalized = proportions / proportions.sum()

        # quantize sample proportions to get preliminary number
        # of samples
        n_samples = (proportions_normalized * n).type(torch.int)

        # catch cases where int conversion results in a number of
        # samples that is different from n and correct them.
        # Current strategy: minimal impact by removing and adding
        # to dominant class
        while ( n_samples.sum() != n ):
            # in case of undershoot, add samples to class
            # with highest number of overall samples
            if n_samples.sum() < n:
                n_samples[np.argmax(proportions_normalized)] += 1

            # in case of overshoot, remove samples from class
            # with highest number of overall samples        
            else:
                n_samples[np.argmax(proportions_normalized)] -= 1
        return n_samples


    def _sample_candidate_voxels(self, slc: Tensor, ground_truth_slice: Tensor, 
                                 n_class_samples: Tensor, seed=None) -> Tensor:
        """ individually sample voxels for each class with samples sizes
            potentially varying among them.

        Parameters
        ----------
        slc : Tensor
            slice from error map, shape n_classes x W x H
            
        ground_truth_slice : Tensor
            ground truth slice, shape n_classes x W x H
            
        n_class_samples : Tensor
            number of samples for each class, shape n_classes
            
        seed : int
            If not None (default), set specified seed
            before sampling.

        Returns
        -------
        samples : Tensor
            mask with samples for specified slice and all classes,
            shape n_classes x W x H
        """

        # seed if specified
        if seed is not None:
            torch.manual_seed(seed)

        # init sampler and output tensor
        sampler = torch.utils.data.WeightedRandomSampler
        samples = torch.zeros_like(slc)

        weights = torch.any(torch.abs(slc).type(torch.uint8), axis=0) * ground_truth_slice

        # print(weights.shape, (weights>0).sum(axis=(1,2)))
        weights = weights / (weights>0).sum(axis=(1,2)).reshape(-1,1,1)
        # iterate over classes, sampling for each independently
        for weight, num_samples, i in zip(weights, n_class_samples, range(slc.shape[0])):
            # catch case where number of samples is zero for a class
            if num_samples > 0:

                # generate uniform weights for false negative voxels
                # weight = (volume > 0) / (volume > 0).sum()
                # upper bound for number of samples to maximum in slice
                max_samples = (weight > 0).sum()
                num_samples = int(min(max_samples, num_samples))
                # 1D coordinates for samples from weight matrix
                #print(num_samples)
                index_list = list(sampler(weight.flatten(), num_samples=num_samples, replacement=False))
                # 2D coordinates for samples from weight matrix
                index_coords = np.unravel_index(index_list, weight.shape)
                # apply mask via coordinates to samples for class i
                samples[i][index_coords] = 1 
        
        return samples


    def _slice_add_neighbors(self, class_samples: Tensor, ground_truth_slice: Tensor) -> Tensor:
        """ creates slice with all sampled interaction candidates and their
            neighborhoods for each class

        Parameters
        ----------
        class_samples : Tensor
            sampled seed mask, shape n_classes x W x H 
    
        ground_truth : Tensor
            ground truth slice, shape n_classes x W x H 


        Returns
        -------
        interaction_mask : Tensor
            mask with added neighbors from brushing,
            shape n_classes x W x H
        """
        interaction_mask           = torch.zeros_like(class_samples, dtype=torch.int64)
        vectorized_binary_erosion  = np.vectorize(binary_erosion,  signature='(j,i),(k,k)->(j,i)')
        vectorized_binary_dilation = np.vectorize(binary_dilation, signature='(j,i),(k,k)->(j,i)')


        for size in self.brush_sizes:
            brush             = torch.ones((size,size))
            brushable_samples = class_samples * torch.tensor(vectorized_binary_erosion(ground_truth_slice, structure=brush))
            brushed_samples   = vectorized_binary_dilation(brushable_samples, structure=brush).astype(int)
            interaction_mask  = torch.bitwise_or(interaction_mask, torch.tensor(brushed_samples))
        
        return interaction_mask

    
    def initial_annotation(self, n_samples: int, paper_init='False') -> Tensor:        
        """ creates the initial annotations. For each direction (saggital, coronal,
            axial), select the slice with the most foreground labels (3 in total).
            (2) For each slice, sample n_samples many seeding points and
                save their position in an annotation mask.
            (3) Apply the largest quadratic brush (from a given range) to each seed
                for which all affected voxels are foreground and add them to
                the annotation mask as well.
            (4) Mask the ground truth labels with the annotation mask and return
        
        Parameters
        ----------
        
        n_samples : int
            number of seed points for each slice. Only needed when 
            paper_init = False
        
        paper_init : bool
            whether the initial annotation matches the annotation used in 
            preliminary paper. Default=False

        Returns
        -------
        interaction_map : Tensor
            shape n_classes x L x W x H

        """

        if paper_init:
            # [("sagittal", 72) -> 72, ("coronal", 87) -> 72, ("axial", 72) -> 72]
            interaction_map = torch.zeros_like(self.gt)
            for orientation in range(3):
                selection                   = [slice(None)] * 4
                if orientation == 0:
                    selection[orientation + 1]  = [73]
                else:
                    selection[orientation + 1]  = [72]
                interaction_map[selection] = self.gt[selection]

            return interaction_map.float()

        elif not paper_init:
            n_classes = self.gt.shape[0]
            #t = self.gt

            inverse_size_weights = self.gt.mean((1,2,3)).sum() / self.gt.mean((1,2,3)).reshape((n_classes,1,1,1))

            t_norm = torch.norm(self.gt  * inverse_size_weights, p=1, dim=0)

            # 1.1) calc sum over l1 norms, e.g. for the l1 norms for segmentation predictions
            slice_sums = self._sum_l1_per_slice(t_norm)

            # 1.2) order slices in descending order by their sum
            axis, indices = self._order_slices_by_sum(slice_sums)
            interaction_map = torch.zeros_like(self.gt, dtype=torch.int64)

            # save data location for later sub - sampling
            data_location = []

            # select one slice for each direction in volume.
            for orientation in range(3):
                #Use the ordered slice list to find best slice for each direction
                selection = [slice(None)] * 4
                index     = indices[np.argmax(axis == orientation)]
                selection[orientation+1] = index
                # select slice from misclassifications and ground truth
                t_selection = self.gt[selection]

                # samples voxels and add their neighborhood to 2D mask
                n_class_samples = self._slice_samples_per_class(t_selection, inverse_size_weights, n_samples)
                print(n_class_samples.sum())
                #print(t_selection.shape, n_class_samples)

                class_samples   = self._sample_candidate_voxels(t_selection, t_selection, n_class_samples=n_class_samples, seed=1)
                brushed_mask    = self._slice_add_neighbors(class_samples, t_selection)
                # make interaction map with same shape as model input
                interaction_map[selection] = torch.bitwise_or(interaction_map[selection], brushed_mask)
                # interaction_map[selection] = ((interaction_map[selection].sum(0) * t_selection) > 0) * 1
                data_location.append((orientation, index))

            return interaction_map.float() #, data_location
    
    
    def refinement_annotation(self, prediction: Tensor, annotation_mask: Tensor, 
                              n_samples: int) -> Tensor:
        """ Finds the slice with the worst prediction across all three axis and 
            annotates parts of it. The annotation happens in multiple steps:
            (1) mask all voxels that are already annotated with annotation_mask
            (2) Sample n_samples many seeding points and save their position in
                an annotation mask
            (3) Apply the largest quadratic brush (from a given range) to each seed
                for which all affected voxels are foreground and add them to
                the annotation mask as well.
            (4) Mask the ground truth labels with the annotation mask and return

        Parameters
        ----------
        prediction : Tensor
            predictions of segmentation model with
            shape n_classes x L x W x H

        annotation_mask : Tensor
            current annotation, shape n_classes x L x W x H

        n_samples : int
            number of samples per slice before brushing

        Returns
        -------
        interaction_map : Tensor
            new annotations, shape n_classes x L x W x H

        """
        n_classes = prediction.shape[0]

        # calculate inverse class frequencies
        inverse_size_weights = self.gt.mean((1,2,3)).sum() / self.gt.mean((1,2,3)).reshape((n_classes,1,1,1))

        # calculate mask for available voxels
        available_voxels = 1 - annotation_mask.float()

        # calculate difference between truth and prediction, i.e. misclassified voxels
        diff = torch.abs(self.gt - prediction.float()) * self.gt * available_voxels
        # print("weights:",inverse_size_weights.flatten())

        # norm over classes weighted by inverse class frequency - importance weight for sampling
        diff_norm = torch.norm(diff  * inverse_size_weights, p=1, dim=0)

        # 1.1) calc sum over l1 norms, e.g. for the l1 norms for segmentation predictions
        slice_sums = self._sum_l1_per_slice(diff_norm)

        # 1.2) order slices in descending order by their sum
        axis, indices = self._order_slices_by_sum(slice_sums)

        # 2.0) select slice with highest importance weight over all axes
        random_selection = np.random.randint(0,6)
        ax  = axis[0]
        slc = indices[0]
        data_location = (ax, slc)
        selection = [slice(None)] + [slice(None)] * 3
        selection[ax + 1] = slc

        # 2.1) calculate number of samples for each class from a raw difference slice
        diff_selection  = diff[selection]
        t_selection     = self.gt[selection]
        n_class_samples = self._slice_samples_per_class(diff_selection, inverse_size_weights, n_samples)
        print(n_class_samples.sum())

        # 2.2) for each class, sample from false negatives as often as specified in n_class_samples
        class_samples = self._sample_candidate_voxels(diff_selection, t_selection, n_class_samples=n_class_samples, seed=1)

        # 2.3) brush all samples with maximum brush from list of brushes
        brushed_mask = self._slice_add_neighbors(class_samples, t_selection)

        # 2.4) create interaction map to return
        interaction_map = torch.zeros_like(self.gt, dtype=torch.int64)
        interaction_map[selection] = torch.bitwise_or(interaction_map[selection], brushed_mask)
        # interaction_map[selection] = ((interaction_map[selection].sum(0) * t_selection) > 0) * 1

        return interaction_map.float() # , data_location # , selection