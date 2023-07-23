import tensorflow as tf

input_shape=(375, 375)
input=tf.keras.layers.Input((375, 375, 3), name="Input")
'''2 VGG Model'''
#2.1 Adding VGG_PreProcess
vgg_pre_process=tf.keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input, name='VGG_PreProcess') (input)
#2.2 Downloading VGG Model
vgg_model=tf.keras.applications.VGG16(include_top=False, input_shape=(375, 375, 3))
vgg_model.trainable=False
vgg=vgg_model (vgg_pre_process)

#2.3 #Adding VGG GAP Layer

'''3 REsNet Model'''
vgg_avg=tf.keras.layers.GlobalAveragePooling2D(name="VGG_Average") (vgg)
#3.1 ResNet PreProcess
resnet_pre_process=tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input, name="Resnet_PreProcess") (input)
#3.2 #Downloading ResNet Model
resnet_model=tf.keras.applications.ResNet50(include_top=False, input_shape=(375, 375, 3))
resnet_model.trainable=False
resnet= resnet_model (resnet_pre_process)
#3.3 #Adding ResNet GAP Layer
resnet_avg=tf.keras.layers.GlobalAveragePooling2D(name="Resent_Average") (resnet)

'''4 Concatinating VGG and ResNet'''
concat=tf.keras.layers.Concatenate( name="Concat") ([vgg_avg, resnet_avg])

#drop1=tf.keras.layers.Dropout(0.5, name="DropOut_1") (concat)
dense1=tf.keras.layers.Dense(512, activation="relu", name="Hidden_1") (concat) #Dense Layer
drop2=tf.keras.layers.Dropout(0.25, name="DropOut_2") (dense1) #DropOut Layer
dense2=tf.keras.layers.Dense(256, activation="relu", name="Hidden_2") (drop2) #Dense Layer
drop3=tf.keras.layers.Dropout(0.25, name="DropOut_3") (dense2) #DropOut Layer
output=tf.keras.layers.Dense(120, activation="softmax", name="Output") (drop3) #Dense Layer(Output)

final_model=tf.keras.models.Model(inputs=[input],
                                  outputs=[output])