Welcome to ConsNet API's documentation!
=======================================

ConsNet API is a part of the `ConsNet <https://github.com/yeliudev/ConsNet>`_ project, it provides several useful functionalities and standard evaluation metrics for the task of human-object interaction (HOI) detection. Currently supported functionalities include but are not limited to:

- Data Loading
   - `Load HICO-DET annotations into a tensor <https://consnet.readthedocs.io/en/latest/consnet.api.data.html#consnet.api.data.load_anno>`_
   - `Convert HICO-DET annotations to COCO format <https://consnet.readthedocs.io/en/latest/consnet.api.data.html#consnet.api.data.convert_anno>`_
- Data Processing
   - `Convert IDs among actions, objects and interactions <https://consnet.readthedocs.io/en/latest/consnet.api.data.html>`_
   - `Compute intersection-over-unions (IoUs) among human-object pairs <https://consnet.readthedocs.io/en/latest/consnet.api.bbox.html#consnet.api.bbox.pair_iou>`_
   - `Perform non-maximum suppression (NMS) among human-object pairs <https://consnet.readthedocs.io/en/latest/consnet.api.bbox.html#consnet.api.bbox.pair_nms>`_
- Standard Evaluation
   - `Evaluate HOI detection results on HICO-DET dataset <https://consnet.readthedocs.io/en/latest/consnet.api.evaluation.html#consnet.api.evaluation.hico_det_eval>`_

.. toctree::
   :caption: Getting Started

   getting_started

.. toctree::
   :caption: API Reference

   consnet.api.data
   consnet.api.bbox
   consnet.api.evaluation

Citation
-------------------------

If you find this project useful for your research, please kindly cite our paper.

.. code-block:: text

   @inproceedings{liu2020consnet,
     title={ConsNet: Learning Consistency Graph for Zero-Shot Human-Object Interaction Detection},
     author={Liu, Ye and Yuan, Junsong and Chen, Chang Wen},
     booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
     pages={4235--4243},
     year={2020}
   }
