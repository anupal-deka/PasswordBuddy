# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'masterpass.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from passwin import Ui_PassWin
import cv2
import numpy as np
import face_recognition
import os

class Ui_Form(object):
    def Verify_face(self):
        path='C:/Python36/PROJECT/_TestApp/image_folder'
    images=[]
    classNames=[]
    mylist=os.listdir(path)

    print(mylist)
    for cl in mylist:
        curImg=cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    print(classNames)
    
    def openWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui=Ui_PassWin()
        self.ui.setupUi(self.window)
        self.window.show()

    def findencoding(images):
        encodelist=[]
        for img in images:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encode=face_recognition.face_encodings(img)[0]
            encodelist.append(encode)
        return encodelist

    def Verify_face(name):
        if name in namelist:
            self.openWindow()

    encodelistKnown=findencoding(images)
    print('encodeing complete')

    cap=cv2.VideoCapture(1)

    while True:
        success,img=cap.read()
        imgs=cv2.resize(img,(0,0),None,0.25,0.25)
        imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

        facesCurFrames=face_recognition.face_locations(imgs)
        enCurrframe=face_recognition.face_encodings(imgs,facesCurFrames)

        for encodeFace,faceloc in zip(enCurrframe,facesCurFrames):
            matches=face_recognition.compare_faces(encodelistKnown,encodeFace)
            faceDis=face_recognition.face_distance(encodelistKnown,encodeFace)
            matchIndex=np.argmin(faceDis)

            if matches[matchIndex]:
                name=classNames[matchIndex].upper()
                y1,x2,y2,x1=faceloc
                y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        cv2.imshow('webcame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow()
            
    def openWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui=Ui_PassWin()
        self.ui.setupUi(self.window)
        self.window.show()

    def messagebox(self,title):
        mess=QtWidgets.QMessageBox()
        mess.setWindowTitle(title)
        mess.setText("Incorrect Password!")
        mess.exec_()

    def checkPass(self):
        if self.lineEdit.text() == 'Anupal':
            self.openWindow()
        else:
            self.messagebox("WARNING!")
          
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(262, 95)
        Form.setMinimumSize(QtCore.QSize(262, 95))
        Form.setMaximumSize(QtCore.QSize(262, 95))
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(30, 20, 198, 18))
        font = QtGui.QFont()
        font.setFamily("Adobe Garamond Pro Bold")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(20, 50, 151, 20))
        self.lineEdit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit.setClearButtonEnabled(True)
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(180, 50, 61, 21))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.checkPass)
        self.pushButton.clicked.connect(self.lineEdit.clear)
        self.pushButton.clicked.connect(Form.close)


        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Master Password"))
        self.label.setText(_translate("Form", "ENTER MASTER PASSWORD"))
        self.pushButton.setText(_translate("Form", "GO!"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())