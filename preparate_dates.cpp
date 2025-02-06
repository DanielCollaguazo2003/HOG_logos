#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;
namespace fs = std::filesystem;

vector<float> calcularHOG(const Mat& imagen) {
    HOGDescriptor hog(
        Size(64, 128),
        Size(16, 16),
        Size(8, 8),
        Size(8, 8),
        9);

    vector<float> descriptores;
    hog.compute(imagen, descriptores);
    return descriptores;
}

void cargarDatos(const string& path, vector<Mat>& imagenes, vector<int>& etiquetas, int etiqueta) {
    for (const auto& entry : fs::directory_iterator(path)) {
        Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (!img.empty()) {
            resize(img, img, Size(64, 128));
            imagenes.push_back(img);
            etiquetas.push_back(etiqueta);
        }
    }
}

int main() {
    vector<Mat> imagenes;
    vector<int> etiquetas;

    cargarDatos("logos/", imagenes, etiquetas, 0);
    cargarDatos("logos/B_bratz", imagenes, etiquetas, 1);
    cargarDatos("logos/B_brother", imagenes, etiquetas, 2);
    cargarDatos("logos/D_dc", imagenes, etiquetas, 3);
    cargarDatos("logos/D_dell", imagenes, etiquetas, 4);
    cargarDatos("logos/Q_qualcomm", imagenes, etiquetas, -1);

    Mat datos;
    Mat etiquetasMat = Mat(etiquetas).reshape(1, etiquetas.size());
    for (const auto& img : imagenes) {
        vector<float> descriptor = calcularHOG(img);
        Mat descriptorMat(descriptor, true);
        datos.push_back(descriptorMat.t());
    }

    FileStorage fs("hog_data.xml", FileStorage::WRITE);
    fs << "data" << datos;
    fs << "labels" << etiquetasMat;
    fs.release();

    cout << "Datos de entrenamiento guardados." << endl;
    return 0;
}
