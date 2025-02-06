#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

// Función para calcular HOG de una imagen
vector<float> calcularHOG(const Mat& imagen) {
    HOGDescriptor hog(
        Size(64, 128), // Tamaño de ventana
        Size(16, 16),  // Tamaño de bloque
        Size(8, 8),    // Paso de bloque
        Size(8, 8),    // Tamaño de celda
        9);            // Número de bins
    
    vector<float> descriptores;
    hog.compute(imagen, descriptores);
    return descriptores;
}

int main() {
    // Cargar el modelo SVM entrenado
    Ptr<SVM> svm = SVM::load("svm_model.xml");
    if (svm.empty()) {
        cerr << "Error al cargar el modelo SVM." << endl;
        return -1;
    }

    // Cargar imagen de prueba
    Mat img = imread("image4.png", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error al cargar la imagen de prueba." << endl;
        return -1;
    }

    // Preprocesamiento mejorado
    Mat img_preprocessed;
    GaussianBlur(img, img_preprocessed, Size(5,5), 0);
    adaptiveThreshold(img_preprocessed, img_preprocessed, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
    
    // Detección de contornos
    vector<vector<Point>> contornos;
    findContours(img_preprocessed, contornos, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // Mostrar contornos detectados
    Mat img_contours;
    cvtColor(img, img_contours, COLOR_GRAY2BGR);
    drawContours(img_contours, contornos, -1, Scalar(0, 255, 0), 2);
    imshow("Contornos Detectados", img_contours);

    vector<Rect> regionesInteres;
    for (const auto& contorno : contornos) {
        Rect bbox = boundingRect(contorno);
        if (bbox.width > 30 && bbox.height > 30) {  // Filtrar regiones pequeñas
            regionesInteres.push_back(bbox);
        }
    }

    // Procesar cada región detectada
    for (const auto& rect : regionesInteres) {
        Mat roi = img(rect);
        resize(roi, roi, Size(64, 128));
        equalizeHist(roi, roi); // Normalizar iluminación
        imshow("ROI", roi);
        waitKey(500);
        
        vector<float> descriptor = calcularHOG(roi);
        Mat descriptorMat(descriptor, true);
        descriptorMat = descriptorMat.t();

        cout << "Tamaño del descriptor: " << descriptorMat.size() << endl;

        int resultado = svm->predict(descriptorMat);

        vector<string> clases = {"Brother", "Bratz", "DC", "DELL", "Qualcomm"};
        string label = (resultado >= 0 && resultado < clases.size() ? clases[resultado] : "Desconocido");

        rectangle(img, rect, Scalar(0, 0, 255), 2);
        putText(img, label, rect.tl(), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    }

    // Mostrar resultado
    imshow("Detección de Logos", img);
    waitKey(0);
    return 0;
}
