# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

#### The process behind converting custom layers involves...
- First need to check for the supported layers in netwrok
- If there is any layer which is not supported yet in that case we have register that layer to IECORE
- After registering we need to optimize the model and then make IR
- We will then use this IR for Inference Engine

#### Some of the potential reasons for handling custom layers are...
When we don't use pretrained model from OpenVINO™ Toolkit zoo, in that case our custom model layers may not be compatible with OpenVINO™ Toolkit supported layers. In that case we need to make use of custom layers in OpenVINO™ Toolkit. For example, if we need to make a different threshold or activation function than supported ones. 

## Comparing Model Performance

#### My method(s) to compare models before and after conversion to Intermediate Representations were...
I compared model performace through model size as I didn't use this particualar without OpenVINO™ Toolkit conversion

#### The size of the model pre- and post-conversion was...

Before conversion .pb file is 68,055 KB
and after conversion .xml file is 110 KB; .bin file is 65,697 KB which is much less then before conversion

## Assess Model Use Cases

#### Some of the potential use cases of the people counter app are...

- Retail store can use for customer demography
- Office Front door can use this office security purpose
- School can use it for student activity monitoring
- Market place can use is it for security

#### Each of these use cases would be useful because...
OpenVINO™ Toolkit has portability and efficiency with durability. We can easily deploy anywhere with great accuracy

## Assess Effects on End User Needs

#### Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows...
Lighting is crucial as models we will use need to draw bounding boxes with detection which is not much robust like MaskRCNN. So lighting can affect detection where aggreagtion of pixels determine the detection.
Without model accuracy we will fail to continuously detect people and it also hamper spped as well. Here we have used threshold probability of .6, but if we degrade then we will loose so many detections as we will not get model accuracy there. Based on focal length we will miss the predictor pixels in our image that's why focal length as well as image size matters

## Model Research

I used ssd_mobilenet_v2_coco_2018_03_29 for my use case which we used in our lessons and the [model source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)

I used the command for conversion 

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
