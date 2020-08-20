import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
import os
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import numpy as np
from keras.preprocessing import image
import sklearn.metrics as metrics
plt.style.use('ggplot')

def generate_mfcc_from_dataset(song,f, file):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfcc, x_axis = 'time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    
    if not os.path.isdir("dataset/training_set/"+f):
        print("creating respective folder")
        os.mkdir("dataset/training_set/"+f)
    print("saving mfcc picture for " + file)
    # photoNumber = str(count)
    plt.savefig("dataset/training_set/"+f+"/"+"{}.png".format(file))
    plt.show()
    plt.close()

    
path_for_genre_music = "genres/"

music_genres_list = ['blues', 'classical', 'country', 'disco', 'hiphop','jazz','metal', 'pop', 'reggae', 'rock']
for f in music_genres_list:
    file_count = sum([len(files) for r, d, files in os.walk(path_for_genre_music+f)])
    print("There are ", file_count , "files in ", f)
    for root, dirs, files in os.walk(path_for_genre_music+f):
        for file in files:
                generate_mfcc_from_dataset(root+"/"+file, f, file)
                
#initializing the CNN
classifierCNN = Sequential()
classifierCNN.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation ='relu' ))
classifierCNN.add(MaxPooling2D(pool_size = (10, 10)))
classifierCNN.add(Flatten())
classifierCNN.add(Dense(units = 128, activation = 'relu'))
classifierCNN.add(Dense(units = 10, activation = 'sigmoid'))
classifierCNN.compile(optimizer = 'adam', loss= 'categorical_crossentropy', metrics = ['accuracy'])  


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory( 'dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64), 
                                            batch_size=32, 
                                            class_mode='categorical')
classifierCNN.fit_generator(training_set, 
                         steps_per_epoch=8000, 
                         epochs=15, 
                         validation_data=test_set, 
                         validation_steps=2)  

training_set.class_indices
#PART 3-  MAKING NEW PREDICTION with Trained model of CNN

# Loading song here
def choosenfile(pathoffile):
    filed = pathoffile
    y, _ = librosa.load(filed)
    mfcc = librosa.feature.mfcc(y)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfcc, x_axis = 'time', y_axis='mel')
    plt.colorbar()
    plt.title(filed)
    plt.tight_layout()
    os.chdir('E:/Sound Processing')
    if not os.path.isdir("output"):
        print("creating output folder")
        os.mkdir("output")
    
    plt.savefig("output"+"/"+"{}.png".format("output"))
    plt.show()
    plt.close()
    test_image = image.load_img('output/'+"output"+".png",target_size = (64 ,64))
    test_image = image.img_to_array(test_image)
    
    test_image = np.expand_dims(test_image,axis = 0)
    result = classifierCNN.predict(test_image)
    print(result[0][0]) 
    if result[0][0] == 0:
        prediction = 'blues'
    elif result[0][0] == 1:
        prediction = 'classical'
    elif result[0][0] == 2:
        prediction = 'country'
    elif result[0][0] == 3:
        prediction = 'disco'
    elif result[0][0] == 4:
        prediction = 'hiphop'
    elif result[0][0] == 5:
        prediction = 'jazz'
    elif result[0][0] == 6:
        prediction = 'metal'
    elif result[0][0] == 7:
        prediction = 'pop'
    elif result[0][0] == 8:
        prediction = 'reggae'
    elif result[0][0] == 9:
        prediction = 'rock'
        
    return prediction
         

# For Creating GUI for music genre Identification
from tkinter import *
from tkinter.filedialog import askopenfilename
import os
from tkinter import messagebox


window = Tk()
window.title("MUSIC GENRE IDENTIFICTION")
titl = Label(window, bg="yellow" , text="MUSIC GENRE IDENTIFICATION", font = ("Arial Bold", 40), )
titl.grid(column=1, row=0)
choose = Label(window, bg="red" , text="Choose your music file for genre identification or Classify genre of different song", font = ("Arial", 15) )
choose.grid(column=1, row=3)
titl = Label(window,  text="" )
titl.grid(column=1, row=5)

def choosen():
    currentdir = os.getcwd()
    name = askopenfilename(initialdir=currentdir,
                        filetypes =(("Music file", "*.wav"),("All Files","*.*")),
                        title = "choose a music file"
                        )
    print(name)
    if name.endswith(".wav"):
        titl = Label(window,  text="" )
        titl.grid(column=1, row=7)
        w = Label(window, bg="yellow", text= "choosen song genre is " + Gotgenre , font = ("Arial", 20))
        w.grid(column = 1, row = 10)
        messagebox.showinfo('Genre Idenified', 'The Genre of the song from path '+name +" is " + Gotgenre)
    else:
        messagebox.showinfo('File extension problem', 'Please choose a music file')
    Gotgenre = choosenfile(name)

btn = Button(window, height=2, width=16, text="Identify Music Genre", bg="sky blue", fg="white", font = ("Arial Bold", 20), command=choosen)
btn.grid(column=1, row=6)
y = Label(window, text= "  ")
y.grid(column = 1, row = 7))
window.geometry('820x400')
window.mainloop()    
    







