#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
    // Cargar datos desde archivo XML
    Mat datos, etiquetas;
    FileStorage fs("hog_data.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error al abrir el archivo de datos." << endl;
        return -1;
    }
    fs["data"] >> datos;
    fs["labels"] >> etiquetas;
    fs.release();

    etiquetas.convertTo(etiquetas, CV_32S); // OpenCV requiere etiquetas en int32

    // Crear y configurar el SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    // Entrenar el modelo SVM
    cout << "Entrenando SVM..." << endl;
    svm->train(datos, ROW_SAMPLE, etiquetas);

    // Guardar el modelo entrenado
    svm->save("svm_model.xml");
    cout << "Modelo SVM entrenado y guardado exitosamente." << endl;
    
    return 0;
}
