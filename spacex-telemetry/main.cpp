/*
* cropped_word_recognition.cpp
*
* A demo program of text recognition in a given cropped word.
* Shows the use of the OCRBeamSearchDecoder class API using the provided default classifier.
*
* Created on: Jul 9, 2015
*     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
*/

#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iomanip>
#include <boost/format.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::text;

struct Telemetry {
    unsigned int fps;
    unsigned long frames;
    std::vector<unsigned int> velocity;

    Telemetry(unsigned int fps, unsigned long frames) 
        : fps(fps), frames(frames)
    {
        velocity.reserve(frames);
    }

    void add(unsigned int velocity) {
        this->velocity.push_back(velocity);
    }

    void write_data(std::ostream& out) {
        for (size_t x = 0; x < velocity.size(); x++) {
            out << x << "," << velocity[x] << endl;
        }
    }
};

Mat* currentFrame;

void onMouse(int event, int x, int y, int, void*)
{
    if (event != CV_EVENT_LBUTTONDOWN)
        return;

    cout << "(" << x << ", " << y << "): " << (int)currentFrame->at<uchar>(y, x) << endl;
}

int count_digits(int n) {
    int digits = 1;
    for (int x = 10; x < 100000; x *= 10) {
        if (n / x > 0) {
            digits++;
        } else {
            break;
        }
    }
    return digits;
}

template <size_t N>
int find_component_wise(tesseract::TessBaseAPI& tess, Mat& velocity)
{
    tess.SetImage((uchar*)velocity.data, velocity.size().width, velocity.size().height, velocity.channels(), velocity.step1());
    Boxa* boxes = tess.GetComponentImages(tesseract::RIL_SYMBOL, true, NULL, NULL);

    char result[N + 1];
    if (boxes->n != N) {
        throw std::runtime_error("Expected and found boxes don't match!");
    }

    printf("Found %d textline image components.\n", boxes->n);
    for (int i = 0; i < boxes->n; i++) {
        BOX* box = boxaGetBox(boxes, i, L_CLONE);
        tess.SetRectangle(box->x, box->y, box->w, box->h);
        char* ocrResult = tess.GetUTF8Text();
        result[i] = ocrResult[0];
        int conf = tess.MeanTextConf();
        fprintf(stdout, "Box[%d]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %s",
                i, box->x, box->y, box->w, box->h, conf, ocrResult);
    }
    result[N] = 0;
    return atoi(result);
}

int main(int argc, char* argv[])
{
    //cv::Mat image = cv::imread("D:\\dev\\spacex-ocr\\spacex-telemetry\\spacex-telemetry\\08893.png");
    //cv::threshold(image, image, 50, 255, cv::THRESH_BINARY);

    std::string file = "E:\\jcsat14.mp4";

    //
    tesseract::TessBaseAPI tess;
    if (tess.Init("D:\\dev\\spacex-ocr\\spacex-telemetry\\spacex-telemetry", "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    tess.SetVariable("tessedit_char_whitelist", "0123456789.");

    VideoCapture cap(file);
    if (!cap.isOpened())  // check if we succeeded
        return -1;

    double fps = cap.get(CAP_PROP_FPS);
    unsigned long frame_count = cap.get(CAP_PROP_FRAME_COUNT);
    int width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

    cout << boost::format("Opened %1% (%2% fps, %3%x%4%)") % file % fps % width % height << endl;

    if (width != 1280 || height != 720) {
        cout << "Expects 720p video" << endl;
        return 1;
    }

    //cap.set(CAP_PROP_POS_MSEC, 1786 * 1000);
    cap.set(CAP_PROP_POS_MSEC, 1686 * 1000);

    namedWindow("JCSAT-14", WINDOW_AUTOSIZE);
    setMouseCallback("JCSAT-14", onMouse, 0);

    //imshow("Display window", image);

    Mat frame, velocity, velocity3, altitude, altitude3;
    Rect velocityRect(1041, 142, 72, 20);
    Rect altitudeRect(1181, 142, 72, 20);
    Rect displayRect(541, 142, 144, 40);
    Rect altitudeDisplayRect(541, 342, 144, 40);

    int prev_digits = 0, prev_velocity = 0;
    Telemetry telemetry(fps, frame_count);

    for (int x = 0; ; ++x)
    {
        cap >> frame;
        currentFrame = &frame;

        Size size = frame.size();
         if (frame.empty())
            break;
        
        frame(velocityRect).copyTo(velocity);
        resize(velocity, velocity, Size(velocity.size().width * 2, velocity.size().height * 2));

        frame(altitudeRect).copyTo(altitude);
        resize(altitude, altitude, Size(altitude.size().width * 2, altitude.size().height * 2));

        cv::cvtColor(velocity, velocity, CV_BGR2GRAY);
        cv::threshold(velocity, velocity, 130, 255, cv::THRESH_BINARY);

        cv::cvtColor(altitude, altitude, CV_BGR2GRAY);
        cv::threshold(altitude, altitude, 130, 255, cv::THRESH_BINARY);

        tess.SetImage((uchar*)velocity.data, velocity.size().width, velocity.size().height, velocity.channels(), velocity.step1());
        tess.Recognize(0);
        int ivelocity = atoi(tess.GetUTF8Text());

        tess.SetImage((uchar*)altitude.data, altitude.size().width, altitude.size().height, altitude.channels(), altitude.step1());
        tess.Recognize(0);
        float faltitude = atof(tess.GetUTF8Text());

        //cout << boost::format("%+5f% v = %+5d%") %  % ivelocity << endl;
        //cout << "Velocity: " << ivelocity << " km/h" << endl;

        int new_digits = count_digits(ivelocity);
        if (new_digits < prev_digits) {
            cout << "Missed some characters" << endl;
            int nvelocity = find_component_wise<5>(tess, velocity);
            new_digits = count_digits(nvelocity);

            cout << "Fix " << ivelocity << " -> " << nvelocity << endl;
            ivelocity = nvelocity;
        }

        
        printf("%.5f, v = %5d, d = %.2f\n", (x / (double)fps), ivelocity, faltitude);

        telemetry.add(ivelocity);

        if (prev_velocity > 9000 && ivelocity < prev_velocity) {
            cv::cvtColor(velocity, velocity3, CV_GRAY2BGR);
            line(velocity3, Point(115, 0), Point(115, velocity.size().height - 1), Scalar(0, 0, 255));
            velocity3.copyTo(frame(displayRect));

            cv::cvtColor(altitude, altitude3, CV_GRAY2BGR);
            altitude3.copyTo(frame(altitudeDisplayRect));

            imshow("JCSAT-14", frame);
            waitKey();
        }

        prev_velocity = ivelocity;
        prev_digits = new_digits;

        if (ivelocity == 26720) {
            break;
        }

        continue;

        
        cv::cvtColor(velocity, velocity3, CV_GRAY2BGR);
        line(velocity3, Point(115, 0), Point(115, velocity.size().height - 1), Scalar(0, 0, 255));
        velocity3.copyTo(frame(displayRect));

        if (prev_velocity == 17 && ivelocity == 11) {
            tess.SetImage((uchar*)velocity.data, velocity.size().width, velocity.size().height, velocity.channels(), velocity.step1());
            tess.Recognize(0);
            cout << atoi(tess.GetUTF8Text()) << endl;
            imshow("JCSAT-14", frame);
            waitKey(); // waits to display frame
        }

        

        imshow("JCSAT-14", frame);
        //waitKey(); // waits to display frame
    }

    //cv::Mat sub = image;// image(cv::Rect(50, 200, 300, 100));
    //tess.SetImage((uchar*)sub.data, sub.size().width, sub.size().height, sub.channels(), sub.step1());
    //tess.Recognize(0);
    //const char* out = tess.GetUTF8Text();

    //cout << out << endl;

    ofstream myfile;
    myfile.open("velocity.csv");
    telemetry.write_data(myfile);

    return 0;
}