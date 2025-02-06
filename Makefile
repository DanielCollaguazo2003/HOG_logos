all:
	g++ Principal.cpp -std=c++17 -I/home/daniel/aplicaciones/Librerias/opencv/opencvi/include/opencv4/ -L/home/daniel/aplicaciones/Librerias/opencv/opencvi/lib -lopencv_core -lopencv_ml -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_features2d -lopencv_objdetect  -o vision.bin
run:
	./vision.bin
