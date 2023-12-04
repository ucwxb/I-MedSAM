# I-MedSAM: Implicit Medical Image Segmentation with Segment Anything

Official code release for [I-MedSAM: Implicit Medical Image Segmentation with Segment Anything](https://arxiv.org/pdf/2311.17081).

<p align='center'>
  <b>
    <a href="https://arxiv.org/pdf/2311.17081">Paper</a>
    |
    <a href="https://github.com/ucwxb/I-MedSAM">Code</a> 
  </b>
</p> 
  <p align='center'>
    <img src='static/I-MedSAM.png' width='1000'/>
  </p>
  <p align='center'>
<video controls>
  <source src="static/video.mp4" type="video/mp4">
</video>
  </p>

**Abstract**: With the development of Deep Neural Networks (DNNs), many efforts have been made to handle medical image segmentation. Traditional methods such as nnUNet train specific segmentation models on the individual datasets. Plenty of recent methods have been proposed to adapt the foundational Segment Anything Model (SAM) to medical image segmentation. However, they still focus on discrete representations to generate pixel-wise predictions, which are spatially inflexible and scale poorly to higher resolution. In contrast, implicit methods learn continuous representations for segmentation, which is crucial for medical image segmentation. In this paper, we propose I-MedSAM, which leverages the benefits of both continuous representations and SAM, to obtain better cross-domain ability and accurate boundary delineation. Since medical image segmentation needs to predict detailed segmentation boundaries, we designed a novel adapter to enhance the SAM features with high-frequency information during Parameter Efficient Fine Tuning (PEFT). To convert the SAM features and coordinates into continuous segmentation output, we utilize Implicit Neural Representation (INR) to learn an implicit segmentation decoder. We also propose an uncertainty-guided sampling strategy for efficient learning of INR. Extensive evaluations on 2D medical image segmentation tasks have shown that our proposed method with only 1.6M trainable parameters outperforms existing methods including discrete and continuous methods. The code will be released.

