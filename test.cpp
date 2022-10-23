#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <algorithm>

struct detectionResult
{
	cv::Rect plateRect;
	double confidence;
	int type;
};

void NMS(std::vector<detectionResult>& vResultRect);

int main() {

	cv::Mat img = cv::imread("dog.jpg");

	using namespace cv::dnn;
	const float confidenceThreshold = 0.24f;
	Net m_net;
	std::string yolo_cfg = "yolov4.cfg";
	std::string yolo_weights = "yolov4.weights";
	m_net = readNetFromDarknet(yolo_cfg, yolo_weights);
	m_net.setPreferableBackend(DNN_BACKEND_OPENCV);
	m_net.setPreferableTarget(DNN_TARGET_CPU);

	cv::Mat inputBlob = blobFromImage(img, 1 / 255.F, cv::Size(416, 416), cv::Scalar(), true, false); //Convert Mat to batch of images

	m_net.setInput(inputBlob);

	std::vector<cv::Mat> outs;

	cv::Mat detectionMat = m_net.forward();

	std::vector<detectionResult> vResultRect;

	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float* prob_array_ptr = &detectionMat.at<float>(i, probability_index);
		size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
		if (confidence > confidenceThreshold)
		{
			float x_center = detectionMat.at<float>(i, 0) * (float)img.cols;
			float y_center = detectionMat.at<float>(i, 1) * (float)img.rows;
			float width = detectionMat.at<float>(i, 2) * (float)img.cols;
			float height = detectionMat.at<float>(i, 3) * (float)img.rows;
			cv::Point2i p1(round(x_center - width / 2.f), round(y_center - height / 2.f));
			cv::Point2i p2(round(x_center + width / 2.f), round(y_center + height / 2.f));
			cv::Rect2i object(p1, p2);

			detectionResult tmp;
			tmp.plateRect = object;
			tmp.confidence = confidence;
			tmp.type = objectClass;
			vResultRect.push_back(tmp);
		}
	}

	NMS(vResultRect);

	for (int i = 0; i < vResultRect.size(); i++)
	{
		cv::rectangle(img, vResultRect[i].plateRect, cv::Scalar(0, 0, 255), 2);
		printf("index: %d, confidence: %g\n", vResultRect[i].type, vResultRect[i].confidence);
	}

	cv::imshow("img", img);
	cv::waitKey();

	return 0 ;
}

void NMS(std::vector<detectionResult>& vResultRect)
{
	for (int i = 0; i < vResultRect.size() - 1; i++)
	{
		for (int j = i + 1; j < vResultRect.size(); j++)
		{
			double IOURate = (double)(vResultRect[i].plateRect & vResultRect[j].plateRect).area() / (vResultRect[i].plateRect | vResultRect[j].plateRect).area();
			if (IOURate >= 0.5)
			{
				if (vResultRect[i].confidence > vResultRect[j].confidence) {
					vResultRect.erase(vResultRect.begin() + j);
					j--;
				}
				else {
					vResultRect.erase(vResultRect.begin() + i);
					i--;
					break;
				}
			}
		}
	}
}