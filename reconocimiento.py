#Clase para el entrenamiento de nuestro modelo
import cv2
import os
import numpy

print('Reconociendo Rostro...')

#Directorio donde se encuentran las carpetas con las fotos de los compañeros de Patrones para el entrenamiento
dir_faces = 'Datasets/att_faces/personas'

#Variable para reducir el tamaño de las fotos de los compañeros para que sean iguales a las del Dataset ATT
size = 4

#Se crea una lista de imagenes y una lista de nombres correspondientes con las fotos de los compañeros
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(dir_faces):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(dir_faces, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (112, 92)

#Se crea una matriz con las dos listas anteriores
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

#Con herramientas de OpenCV se entrena el modelo a partir de las listas anteriores
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)


#Se Utiliza el modelo entrenado para la prediccion
face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')

#En la variable ruta se envia la direccion de la foto del compañero que se quiere reconocer
#Foto de prueba Fabian Alfaro: 0.jpeg
#Foto de prueba Warner Cortes: 1.jpg

ruta="warner.jpg"
img=cv2.imread(ruta)

#Se aplica un filtro blanco y negro y se redimensiona la foto para favorecer a la precision del modelo
gray = cv2.imread(ruta,0)
mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

#Se obtienen las coordenadas del rostro en la foto
faces = face_cascade.detectMultiScale(mini)
for i in range(len(faces)):
    face_i = faces[i]
    (x, y, w, h) = [v * size for v in face_i]
    face = gray[y:y + h, x:x + w]
    face_resize = cv2.resize(face, (im_width, im_height))

    #Se hace una prediccion para el reconocimiento del rostro
    prediction = model.predict(face_resize)

     #Se dibuja un rectangulo en las coordenadas del rostro para enmarcarlo
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #Variable que tendra el nombre de la persona reconocida en la foto
    cara = '%s' % (names[prediction[0]])

    #Si la prediccion tiene una precision menor a 100 se toma como prediccion valida
    if prediction[1]<100 :
      #Se escribe el nombre de la persona que reconocio
      cv2.putText(img,'%s - %.0f' % (cara,prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

    #Si la prediccion es mayor a 100 NO es un reconomiento con la presicion suficiente por lo tanto se determina como un desconocido
    elif prediction[1]>101 and prediction[1]<500:
        #Se escribe desconocido en el rostro detectado
        cv2.putText(img, 'Desconocido',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

    #Se muestra la foto con el reconocimiento obtenido
    cv2.imshow('Reconocimiento facial con PCA y LBP', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Para cerrar el programa con la tecla ESC
key = cv2.waitKey(10)
if key == 27:
    cv2.destroyAllWindows()
