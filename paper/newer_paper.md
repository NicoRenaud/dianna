---
title: 'DIANNA: Deep Insight And Neural Network Analysis'
tags:
  - Python
  - explainable AI
  - Deep Neural Networks
  - ONNX
  - benchmark datasets
authors:
  - name: Elena Ranguelova 
    orcid: 0000-0002-9834-1756
    affiliation: 1
  - name: Christiaan Meijer
    orcid: 0000-0002-5529-5761
    affiliation: 1
  - name: Leon Oostrum
    orcid: 0000-0001-8724-8372
    affiliation: 1
  - name: Yang Liu
    orcid: 0000-0002-1966-8460
    affiliation: 1
  - name: Patrick Bos 
    orcid: 0000-0002-6033-960X
    affiliation: 1
  - name: Giulia Crocioni 
    orcid: 0000-0002-0823-0121
    affiliation: 1
  - name: Matthieu Laneuville 
    orcid: 0000-0001-6022-0046
    affiliation: 2
  - name: Bryan Cardenas Guevara
    orcid: 0000-0001-9793-910X
    affiliation: 2
  - name: Rena Bakhshi 
    orcid: 0000-0002-2932-3028 
    affiliation: 1   
  - name: Damian Podareanu
    orcid: 0000-0002-4207-8725
    affiliation: 2
affiliations:
 - name: Netherlands eScience Center, Amsterdam, the Netherlands
   index: 1
 - name: SURF, Amsterdam, the Netherlands
   index: 2
date: 22 March 2022 # put date here
bibliography: paper.bib
---

# Summary

Researchers often need to understand the mechanisms of artificial intelligence (AI) based insights to judge whether the AI algorithms are valid beyond irrelevant statistical correlations in potentially biased datasets. DIANNA (Deep Insight And Neural Network Analysis) provides a Python-based, user-friendly, and uniform interface to several XAI (eXplainable Artificial Intelligence) methods. Texts, audio tracks or images that have been classified by an AI model can be analyzed and saliency heat-mapped by one or more of several explainability algorithms in DIANNA implementation. DIANNA is designed for all researchers, including non-AI experts, so it contains several ways to interface with the code including a dashboard, which allows code-free implementation of user picked explainability algorithms. DIANNA has been used in research on topics from the humanities to medicine to improve the trustworthiness of AI models. 

# Statement of need

AI systems have been increasingly used in a wide variety of fields, including such data-sensitive areas as healthcare [@alshehrifatima], renewable energy [@kuzlumurat], supply chain [@toorajipourreza] and finance. Automated decision-making and scientific research standards require reliability and trust of the AI technology [@xuwei]. But oftentimes determination of whether an artificial intelligence (AI) system is reliable, fair or even accurate beyond a limited set of data is impossible without insight into the mechanisms of its decision making. XAI is a group of methods and tools that allow human insight into the decision-making processes of AI algorithms. In AI-enhanced research, scientists need to be able to trust high-performing, but opaque AI models used for automation of their data processing pipelines, and potentially XAI algorithms can allow these models to be more transparent.  In addition, XAI has the potential for helping any scientist to "find new scientific discoveries in the analysis of their data” [@hey]. Furthermore, tools for supporting repeatable science are in high demand [@Feger2020InteractiveTF]. 

There are numerous Python XAI libraries, many of which are listed in the Awesome explainable AI [@awesomeai] repository, but none of which solve all problems of XAI research. Popular and widely used packages are Pytorch [@pytorch], LIME [@ribeirolime], Captum [@kokhlikyan2020captum], Lucid [@tflucid], SHAP [@lundbergshap], InterpretML [@nori2019interpretml], PyTorch CNN (Convolutional Neural Networks) visualizations [@uozbulak_pytorch_vis_2021] Pytorch GradCAM [@jacobgilpytorchcam], Deep Visualization Toolbox [@yosinski-2015-ICML-DL-understanding-neural-networks], ELI5 [@eli5]. However, these libraries have limitations that complicate adoption by scientific communities. 

There are limitations of previous work that are library specific. While libraries such as SHAP, LIME, Pytorch GradCAM. have gained great popularity, their methods are not always suitable for the research task and/or data modality. For example, GradCAM is applicable only to images. Most importantly, each library in that class addresses AI explainability with a different method, complicating comparison between methods. Many XAI libraries support a single deep neural net (DNN) format e.g. Lucid supports only TensorFlow [@tf], and Captum - PyTorch [@pytorch] and iNNvestigate [@innvestigatenn] are aimed at Keras users exclusively. Pytorch GradCAM supports a single method for a single format and Convolutional Neural Network Visualizations even limits the choice to a single DNN type. Tools that support a single framework are not "future-proof". For instance, Caffe [@jia2014caffe] was the most popular framework in the computer vision (CV) community in 2018, but it has since been abandoned. ELI5 supports multiple frameworks/formats and XAI methods, but it is unclear how the selection of these methods was made. Furthermore, the library has not been maintained since 2020, so any methods in the rapidly changing XAI field proposed since then are missing. The Deep Visualization Toolbox requires DNN knowledge and is only used by AI experts mostly within the CV community.  

Beyond the specific limitations of existing libraries, on a more fundamental level, the results of previous XAI research did not help to make the technology understandable and trustworthy for non (X)AI experts.  

DIANNA is an open source XAI Python package with the following key characteristics: 

- **Systematically chosen diverse set of XAI methods.**  We have used a relevant subset of the thorough objective and systematic evaluation criteria defined in [@peterflatch]. Several complementary and model-architecture agnostic state-of-the-art XAI methods have been chosen and included in DIANNA [@ranguelova_how_2022].
- **Multiple data modalities.** DIANNA supports images and text, we will extend the input data modalities to embeddings, time-series, tabular data and graphs. This is particularly important to scientific researchers, whose data are in domains different than the  classical examples from CV and natural language processing communities.
- **Open Neural Network Exchange (ONNX) format.** ONNX is the de-facto standard format for neural network models. Not only is the use of ONNX very beneficial for interoperability, enabling reproducible science, but it is also compatible with runtimes and libraries designed to maximize performance across hardware. To the best of our knowledge, DIANNA is the first and only XAI library supporting ONNX.
- **Simple, intuitive benchmark datasets.** We have proposed two new datasets which enable systematic research of the properties of the XAI methods' output and understanding on an intuitive level: Simple Geometric Shapes [@oostrum_leon_2021_5012825] and LeafSnap30 [@ranguelova_elena_2021_5061353]. The classification of tree species on LeafSnap data is a great example of a simple scientific problem tackled with both classical CV and a deep learning method, where the latter outperforms, but needs explanations.  DIANNA also uses well-established benchmarks: a simplified MNIST with 2 distinctive classes only and the Stanford Sentiment Treebank [@socher-etal-2013-recursive].
- **User-friendly interfaces.** DIANNA wraps all XAI methods with a common API and includes a drag-and-drop dashboard.
- **Modular architecture, extensive testing and compliance with modern software engineering practices.** It is very easy for new XAI methods which do not need to access the ONNX model internals to be added to DIANNA. For relevance-propagation type of methods, more work is needed within the ONNX standard [@levitan_onnx_2020] and we hope our work will boost the development growth of ONNX (scientific) models. We welcome the XAI research community to contribute to these developments via DIANNA.
- **Thorough documentation.** The package includes user and developer documentation. It also provides instructions for conversion between ONNX and Tensorflow, Pytorch, Keras or Scikit-learn.

# Key Features

![High level architecture of DIANNA](https://user-images.githubusercontent.com/3244249/158770366-a624d1e0-2eae-43cc-aeb5-bfa33b50b3e4.png)

# Ongoing research using DIANNA 

DIANNA is currently used in the "Recognizing symbolism in Turkish television drama" project [@turkishdrama] to increase insight into the AI models in order to explore how to improve them. DIANNA is also currently used in the "Visually grounded models of spoken language" project, which builds on earlier work from [@chrupala17representations;@alishahi+17;@chrupala18;@chrupala+19]. The goal of this work is a multi-modal model by projecting image and sound data into a common embedded space. Within DIANNA, we are developing XAI methods to visualize and explain these embedded spaces in their complex multi-modal network contexts. Finally, DIANNA was also used in the EU-funded Examode medical research project [@bryancardenas]. Examode deals with very large data sets and since it aims to support physicians in their decision-making, it needs transparent and trustworthy models. 

# Acknowledgements

This work was supported by the [Netherlands eScience Center](https://www.esciencecenter.nl/) and [SURF](https://www.surf.nl/en).

# References

[//]: # "All the refs need to be put in paper.bib file ([open PR #241](https://github.com/dianna-ai/dianna/pull/241)) and cited above using this notation: [@bibentry]."
