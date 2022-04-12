#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>

using namespace cv;
using namespace std;

// Step 1: complete gradient and threshold
// Step 2: complete sobel
// Step 3: complete canny (recommended substep: return Max instead of C to check it) 
// Step 4 (facultative, for extra credits): implement a Harris Corner detector

// Raw gradient. No denoising
void gradient(const Mat&Ic, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	G2 = Mat(m, n, CV_32F);

        float dx = 0.f;
        float dy = 0.f;

        for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {

                    // Compute squared gradient (except on borders)

                    if (i < m-1){
                        dx = I.at<uchar>(i+1,j) - I.at<uchar>(i,j);
                    }

                    if (j < n-1){
                        dy = I.at<uchar>(i,j+1) - I.at<uchar>(i,j);
                    }

                    G2.at<float>(i, j) = pow(pow(dx,2)+pow(dy,2),0.5);
		}
	}
}

// Gradient (and derivatives), Sobel denoising
void sobel(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);

	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G2 = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {

                    if ((i >= 2) && (j >= 2)){
                        Ix.at<float>(i, j) = 1./8.*(-I.at<uchar>(i,j) + I.at<uchar>(i,j-2) -2.*I.at<uchar>(i-1,j) + 2.*I.at<uchar>(i-1,j-2) -I.at<uchar>(i-2,j) + I.at<uchar>(i-2,j-2));

                        Iy.at<float>(i, j) = 1./8.*(I.at<uchar>(i,j) + 2.*I.at<uchar>(i,j-1) + I.at<uchar>(i,j-2) -I.at<uchar>(i-2,j) -2.*I.at<uchar>(i-2,j-1) -I.at<uchar>(i-2,j-2));
                    }

                        G2.at<float>(i, j) = pow( pow(Ix.at<float>(i, j) , 2) + pow(Iy.at<float>(i, j),2) , 0.5);
		}
	}
}

// Gradient thresholding, default = do not denoise
Mat threshold(const Mat& Ic, float s, bool denoise = false)
{
	Mat Ix, Iy, G2;

	if (denoise)
		sobel(Ic, Ix, Iy, G2);
	else
		gradient(Ic, G2);

	int m = Ic.rows, n = Ic.cols;
	Mat C(m, n, CV_8U);
        float Gij;

        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){

                Gij = G2.at<float>(i,j);

                if (Gij > s){
                    C.at<uchar>(i, j) = Gij;
                }

                else{
                    C.at<uchar>(i,j) = 0.;
                }
            }
        }

	return C;
}

bool find_if_max(float direction, int i, int j, Mat G2){

    int m = G2.rows, n = G2.cols;
    float pi = 3.14;
    float Gij = G2.at<float>(i,j);

    float d1 = fabs(direction - pi/2.);
    float d2 = fabs(direction + pi/2.);
    float d3 = fabs(direction - pi/4.);
    float d4 = fabs(direction + pi/4.);
    float d5 = fabs(direction);

    float min_d = min(d1,min(d2,min(d3,min(d4,d5))));

    //direction verticale
    if(min_d == d1 || min_d == d2){

        if (G2.at<float>(i+1,j) > Gij){
            return false;
        }
        if (G2.at<float>(i-1,j) > Gij){
            return false;
        }
        return true;
    }

    //direction horizontale
    else if(min_d == d5){

        if (G2.at<float>(i,j+1) > Gij){
            return false;
        }
        if (G2.at<float>(i,j-1) > Gij){
            return false;
        }
        return true;
    }

    //diagonale montante
    else if(min_d == d3){

        if (G2.at<float>(i-1,j+1) > Gij){
            return false;
        }
        if (G2.at<float>(i+1,j-1) > Gij){
            return false;
        }
        return true;
    }

    //diagonale descendante
    else if(min_d == d4){

        if (G2.at<float>(i+1,j+1) > Gij){
            return false;
        }
        if (G2.at<float>(i-1,j-1) > Gij){
            return false;
        }
        return true;
    }

    return true;
}

// Canny edge detector, with thresholds s1<s2
Mat canny(const Mat& Ic, float s1, float s2)
{
	Mat Ix, Iy, G2;
	sobel(Ic, Ix, Iy, G2);

	int m = Ic.rows, n = Ic.cols;
	Mat Max(m, n, CV_8U);	// Binary black&white image with white pixels when ( G2 > s1 && max in the direction of the gradient )
	// http://www.cplusplus.com/reference/queue/queue/
	queue<Point> Q;			// Enqueue seeds ( Max pixels for which G2 > s2 )
        float direction;
        bool est_max;

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {

                    if(G2.at<float>(i,j) > s1){

                        direction = atan2(Iy.at<float>(i,j) , Ix.at<float>(i,j));
                        est_max = find_if_max(direction,i,j,G2);

                        if (est_max){

                            Max.at<uchar>(i,j) = 255;

                            if (G2.at<float>(i,j) > s2){
                                Q.push(Point(j,i));
                            }

                        }

                        else {
                            Max.at<uchar>(i,j) = 0;
                        }
			// ...
			// ...
			// ...
			// ...
			// if (???)
			//		Q.push(point(j,i)) // Beware: Mats use row,col, but points use x,y
			// Max.at<uchar>(i, j) = ...
                    }

                    else {
                        Max.at<uchar>(i,j) = 0;
                    }
		}
	}

	// Propagate seeds
        Mat C(m, n, CV_8U);
	C.setTo(0);
	while (!Q.empty()) {
		int i = Q.front().y, j = Q.front().x;
		Q.pop();
                // ...ajoute les weaks edges voisines ?
                C.at<uchar>(i,j) = 255;
                if (Max.at<uchar>(i-1,j) == 255){
                    if (C.at<uchar>(i-1,j) == 0.){
                        Q.push(Point(j,i-1));
                    }
                    C.at<uchar>(i-1,j) = 255;
                }
                if (Max.at<uchar>(i+1,j) == 255){
                    if (C.at<uchar>(i+1,j) == 0.){
                        Q.push(Point(j,i+1));
                    }
                    C.at<uchar>(i+1,j) = 255;
                }
                if (Max.at<uchar>(i-1,j-1) == 255){
                    if (C.at<uchar>(i-1,j-1) == 0.){
                        Q.push(Point(j-1,i-1));
                    }
                    C.at<uchar>(i-1,j-1) = 255;
                }
                if (Max.at<uchar>(i,j-1) == 255){
                    if (C.at<uchar>(i,j-1) == 0.){
                        Q.push(Point(j-1,i));
                    }
                    C.at<uchar>(i,j-1) = 255;
                }
                if (Max.at<uchar>(i+1,j-1) == 255){
                    if (C.at<uchar>(i+1,j-1) == 0.){
                        Q.push(Point(j-1,i+1));
                    }
                    C.at<uchar>(i+1,j-1) = 255;
                }
                if (Max.at<uchar>(i-1,j+1) == 255){
                    if (C.at<uchar>(i-1,j+1) == 0.){
                        Q.push(Point(j+1,i-1));
                    }
                    C.at<uchar>(i-1,j+1) = 255;
                }
                if (Max.at<uchar>(i,j+1) == 255){
                    if (C.at<uchar>(i,j+1) == 0.){
                        Q.push(Point(j+1,i));
                    }
                    C.at<uchar>(i,j+1) = 255;
                }
                if (Max.at<uchar>(i+1,j+1) == 255){
                    if (C.at<uchar>(i+1,j+1) == 0.){
                        Q.push(Point(j+1,i+1));
                    }
                    C.at<uchar>(i+1,j+1) = 255;
                }
	}

        return C;
        return Max;
}

// facultative, for extra credits (and fun?)
// Mat harris(const Mat& Ic, ...) { ... }

int main()
{
	Mat I = imread("../road.jpg");

        imshow("Input", I);
	imshow("Threshold", threshold(I, 15));
	imshow("Threshold + denoising", threshold(I, 15, true));
	imshow("Canny", canny(I, 15, 45));
	// imshow("Harris", harris(I, 15, 45));

	waitKey();

	return 0;
}
