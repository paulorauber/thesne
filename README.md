About
-----
This project is intended as a flexible implementation of t-SNE [1] and dynamic
t-SNE [2]. 

The t-SNE cost function is defined symbolically and automatically translated 
into efficient (CPU or GPU) code using theano. Because the derivatives are also 
computed automatically, testing alternative cost functions becomes very easy. 

Dynamic t-SNE is an adaptation of t-SNE for sequences of time-dependent
datasets. It introduces a controllable trade-off between temporal coherence
and projection (embedding) reliability. For more details, please see [2].

If your use of this code results in a publication, please cite (at least) the
original paper by Laurens van der Maaten [1].

This implementation is not nearly as computationally efficient as some
alternatives (e.g., Barnes-Hut t-SNE [3]).

Examples
--------
See the examples directory. Remember to add the directory that contains thesne 
to the PYTHONPATH.

References
----------
[1] Van der Maaten, Laurens and Hinton, Geoffrey. Visualizing data using t-SNE.
    Journal of Machine Learning Research, 2008.

[2] Rauber, Paulo E., Falcão, Alexandre X., Telea, Alexandru C. Visualizing 
    Time-Dependent Data Using Dynamic t-SNE. Proc. EuroVis Short Papers, 2016. 

[3] Van der Maaten, Laurens. Accelerating t-SNE using Tree-Based Algorithms.
    Journal of Machine Learning Research, 2014.