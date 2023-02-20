# AISda
Fish recognition using AI

## A bit of history

Hundreds of thousands of years ago, people have already been impacting wild animals and plants. Species dying off is not a recent phenomenon. And our capacity to change landscapes and animal populations have been made so easy with technology such as fire.

It’s no coincidence that large animals have gone extinct the moment modern humans have intruded into their territory. Humans have been impacting the earth ever since.

Woolly mammoths have been killed off by humans just four thousands of years ago. Neither is it coincidence that giant sloths of Australia have suddenly gone extinct at the same time modern humans stepped foot on the continent. 

## Mass extinction

Today, we have advanced so much that we have started a mass extinction event akin to what happened 66 million years ago.

Our machines like fishing ships, tree cutters, and guns have made killing much more efficient. Our industries and use of fossil fuels have generated so much greenhouse gases to the point of altering the world’s climate. We have invented a material called plastic that doesn’t degrade, goes to our oceans and traps marine life.

An estimated of 200-2000 species go extinct each year.

Meanwhile, 15 billion trees are cut each year. This is equivalent to forests the area of three football fields every second.

## Fish are important

<img src = 'https://live.staticflickr.com/3726/33470628350_bd16d224e1_b.jpg' height="300" align='center'>

Fish are important players in ecosystems. They are food sources for many species down the food chain like birds, seagulls, polar bears, and humans. In the Philippines, more than 50% of our animal protein intake comes from fish. They also play a role in enriching coral reefs with nutrients for them to thrive. Coral reefs are important habitats for marine animals. Coral reefs in the Philippines, in particular, are the most biodiverse in the world.

However, fish populations are now declining due to overfishing and climate change. Their numbers have already halved since 1970. It is thus important that we restore and conserve fish populations. We don’t want to tip the balance of ecosystems on which we rely on.

While fish in the oceans have been declining, AI research, especially in the fields of deep learning, has been gaining traction in the past few years. 

## Trends in AI

Image recognition, in particular, has seen a leap in performance in 2012 when Alex Krizhevsky and his team unveiled a deep convolutional network called AlexNet. In 2015, Microsoft’s AI beat humans in object identification.

<img src = 'media/alexnet.png' height="300" align='center'>

**Our question is how can we use these advancing AI technologies to help in conserving fish in the oceans?**

There are over 30,000 fish species identified so far. Currently, auditing of fish entails divers and manually counting and identifying fish. With the sheer number of fish and large swaths of coral reefs, manual auditing is not scalable. If we can train an AI to identify and count fish from photos and videos, then we can automate and scale auditing using photos captured from underwater cameras. We can expand the scope of audit spatially and in time. This will not replace manual audits by divers but will rather augment them.

UP Marine Science Institute recently unveiled ARAICoBeH (A Rapid Assessment Instrument for Coastal Benthic Habitats) to capture images of underwater habitats. We can tap them to train an AI on their images to automate counting and identification of marine life.

## State-of-the-art in image recognition

<img src = 'https://upload.wikimedia.org/wikipedia/commons/5/5e/Activated_Visual_Cortex_of_Normal_Brain_Versus_Blind_Person%27s_Brain.svg' height="300" align='center'>

The state of the art in image recognition is a deep learning architecture called convolutional neural networks or CNN. This architecture is most apt for our problem of fish recognition.

CNN was inspired by the brain’s visual system located in the visual cortex. Light passes through our eyes and get focused on our retina which then converts those signals into electrochemical signals that our brain can process. Each type of signal stimulates certain neurons. 

For example, signals for circles stimulate a set of neurons for circles, while signals for lines simulate another set of neurons for lines. These shapes and lines that make up an image are called features. Models for various features in the image overlap with each other until you form the entire image. That is what inspired the CNN architecture.

CNN is an automatic way of creating these “filters” very much like polarized filters in a camera. The objective is to extract features from the image that when taken together serve as a DNA of the image.

Here is an example of the features extracted from an image. Each box is a filter for a specific feature. Some filters find circles, some find lines and others boxes. In CNN terms, these filters are called convolution layers.

<p align='center'>
<img src = 'media/filters.png' height="300">
</p>
Here’s an illustration of a CNN architecture. The input is an image. The output is the label of that image, which in this case is a robot. The filters are each of the boxes in between the input and the output. The way this works is an image passes through many filters that select certain features. These features cumulatively make up a feature map, which again is the DNA of the image. We can then use this DNA to identify the image.

<p align='center'>
  <img src = 'https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png' height="300">
</p>

Initially, the filters are random and no feature is extracted from an image. But by showing the model many examples of images, CNN automatically finds the optimal filters that will best identify objects from those images.

Once we have a trained model, the next question is how do we find an object in an image?

<p align='center'>
  <img src = 'media/clownfish-gif-bruteforce.gif' height="300">
</p>

One way to do it is to have a box scan an image from left to right and then top-down. We can then run each scan in our CNN model and try to predict what’s in the image.

Most of the time, there will be no object in the box, but we will keep scanning until an object fits inside the box.

The problem with this approach is that it's a brute force approach. We will use a lot of computation scanning and running each scan in our CNN model.

## YOLO

A new, faster approach is YOLO. YOLO, in this case, doesn’t stand for You Only Live Once but rather You Only Look Once.
Published in 2015, this is hundreds of times faster than the brute force approach. Because the model literally looks at all potential boxes once.

Many improvements have been incorporated in the model ever since. In this project, we used the YOLO v2 algorithm published in December 2016.

YOLO v2 works this way: first, the image is divided into a grid, shown here by the color green. In our case, we divided the image into a 3X3 grid equivalent to 9 cells. For each cell, the model predicts n bounding boxes. In our simplified example, we predicted two bounding boxes for each cell. Finally, the model predicts the object in each bounding box with a certain confidence. 

<p align='center'>
  <img src = 'media/clownfish-gif-yolo.gif' height="300">
</p>

In detail, YOLO predicts the x and y coordinates of the center, the width, and height of each bounding box, and the probability that an object exists in a box. Finally, if an object exists in the box, the model predicts the probability the object belongs to a class. 

All in all, if we follow the math, the model is trained to predict five bounding box properties plus the number of classes in our dataset.

<p align='center'>
  <img src = 'media/how yolo works.png' height="300">
</p>

The confidence of a box is the likelihood it contains an object. If it exceeds a threshold, we retain that bounding box and we identify the object.

Because you only look once, YOLO is hundreds of times faster than competing models and can identify thousands of objects in real-time. This makes it perfect for applications that require counting many objects in an image over long periods of time.
In our project, we wanted to see a proof of concept of using YOLO for quick fish counting and identification.

## Method

<p align='center'>
  <img src = 'media/model flowchart.png' height="300">
</p>

As a proof of concept, we used footage of fish we captured in an Ocean Park.
We then split the footage into training and test set. The first 75% of the footage was used for training, while the remaining 25% was used for the test.

To improve performance, we needed to increase the number of training data. We doubled the training data by adding flipped images. All in all, we used 226 images for training. Each fish in each image was manually annotated with their bounding box coordinates, dimensions, and fish species.

The baseline YOLO v2 model has 19 convolutional layers, which we mentioned earlier are the filters. In our case, we trained on a smaller YOLO model with only 9 convolutional layers. We built up our model over pre-trained weights that were trained using the Visual Object Challenge 2012.

We selected only three species for training: the blue tang surgeonfish, damselfish, and butterflyfish.

<p align='center'>
  <img src = 'media/species.png' height="300">
</p>

With the help of volunteers, each image in the training set was manually annotated using the online tool makesense.ai and a custom python script. The output was an XML file for each image that contains the bounding box coordinates, width, height, and fish species.

A total of 700+ butterflyfish, 600+ blue tang surgeonfish, 2000+ damselfish were annotated in 200 images.

## AIsda

<p align='center'>
  <img src = 'media/Aisda-gif.gif' height="300">
</p>

This is a video showing the prediction of our quick model on the test set. The model is promising in that it is able to identify and count fish correctly most of the time. 

Currently, this is just a proof of concept and the capacity to predict is limited to this particular scene and three species of fish.

We tested the model on a different scene taken from Youtube. The model correctly predicts the fish species but misses creating bounding boxes on the fish. This is due to the limited data the model was trained on.

<p align='center'>
  <img src = 'media/fish_test_1.gif' height="300">
</p>

Moving forward, we will train on more fish species in more varied environments.
We will also experiment on various architectures and hyperparameters based on the YOLO model and find what best predicts fish.

Our vision is a reliable fish recognition software that will be used to monitor fish populations scalable across the country.

## Acknowledgements

We would like to thank Microsoft for the cloud computing grant through their AI for Earth Grant.

We would like to give special thanks to our volunteers: Frances Claravall, Reynier Tasico, and Kevin Kaw for annotating fish images. Thanks to Kathy Gavile for her advice on the fish problem.

Tensorflow implementation of YOLO: https://github.com/thtrieu/darkflow
