# include<opencv2/imgcodecs.hpp>
# include<opencv2/highgui.hpp>
# include<opencv2/imgproc.hpp>
# include<opencv2/objdetect.hpp>
# include<iostream>

using namespace cv;
using namespace std;

void main() {
	string path = "DS-IQ-003-PixelVariation-Video.mp4";
	VideoCapture cap(path);
	Mat img;

	while (true) {
		cap.read(img);
		vector<Mat> bgr_planes;
		split(img, bgr_planes);

		int histSize = 256;

		float range[] = { 0, 256 };
		const float* histRange = { range };


		Mat b_hist, g_hist, r_hist;

		calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, true, false);
		calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, true, false);
		calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, true, false);
		
		int hist_w = 400; int hist_h = 400;
		int bin_w = cvRound((double)hist_w / histSize);

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

		normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		for (int i = 1; i < histSize; i++)
		{
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),Scalar(255, 0, 0), 2, 8, 0);
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),Scalar(0, 255, 0), 2, 8, 0);
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),Scalar(0, 0, 255), 2, 8, 0);
		}

		imshow("Pixel Variation", histImage);
		imshow("Video", img);
		waitKey(30);
	}
}