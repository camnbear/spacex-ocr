/*
* cropped_word_recognition.cpp
*
* A demo program of text recognition in a given cropped word.
* Shows the use of the OCRBeamSearchDecoder class API using the provided default classifier.
*
* Created on: Jul 9, 2015
*     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
*/

#include <opencv2/text.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iomanip>
#include <boost/format.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::text;

struct TelemetryPoint {
    unsigned int velocity;
    unsigned int altitude;

    TelemetryPoint(unsigned int velocity, unsigned int altitude)
        : velocity(velocity), altitude(altitude)
    {}
};

struct Telemetry {
    unsigned int fps;
    unsigned long frames;
    std::vector<TelemetryPoint> data;

    Telemetry(unsigned int fps, unsigned long frames) 
        : fps(fps), frames(frames)
    {
        data.reserve(frames);
    }

    void add(TelemetryPoint point) {
        data.push_back(std::move(point));
    }

    void write_data(std::ostream& out) {
        for (size_t x = 0; x < data.size(); x++) {
            const TelemetryPoint& point = data[x];
            out << x << "," << point.velocity << "," << point.altitude << endl;
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
    for (int x = 10; ; x *= 10) {
        if (n / x > 0) {
            digits++;
        } else {
            break;
        }
    }
    return digits;
}

class SpaceXOCR {
private:
    tesseract::TessBaseAPI tess;
    VideoCapture cap;
    cv::Mat frame;
    
    double fps;
    unsigned long frame_count;
    unsigned int width, height;

    void configure_for_velocity() {
        tess.SetVariable("tessedit_char_whitelist", "0123456789");
    }

    void configure_for_altitude() {
        tess.SetVariable("tessedit_char_whitelist", "0123456789.");
    }

    template <typename T> T as_number(const char* str);

    template <>
    int as_number(const char* str) {
        return atoi(str);
    }

    template <>
    double as_number(const char* str) {
        return atof(str);
    }

    template <typename T>
    T recognize(const cv::Mat& frame, const cv::Rect& roi, cv::Mat& mat)
    {
        resize(frame(roi), mat, roi.size() * 2, 0.0, 0.0, cv::INTER_CUBIC);
        cv::cvtColor(mat, mat, CV_BGR2GRAY);

        // accuracy is very dependent on this threshold value
        cv::threshold(mat, mat, 130, 255, cv::THRESH_BINARY);

        tess.SetImage((uchar*)mat.data, mat.size().width, mat.size().height, mat.channels(), mat.step1());
        tess.Recognize(0);

        return as_number<T>(tess.GetUTF8Text());
    }

    int find_component_wise(cv::Mat& mat)
    {
        tess.SetImage((uchar*)mat.data, mat.size().width, mat.size().height, mat.channels(), mat.step1());
        Boxa* boxes = tess.GetComponentImages(tesseract::RIL_SYMBOL, true, NULL, NULL);

        char result[64];
        if (boxes->n >= 64) {
            throw std::runtime_error("Too many boxes..");
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
        result[boxes->n] = 0;
        return atoi(result);
    }

    void display(cv::Mat& velocity, cv::Rect& velocity_rect, cv::Mat& altitude, cv::Rect& altitude_rect) {
        cv::Mat velocity3, altitude3;

        cv::cvtColor(velocity, velocity3, CV_GRAY2BGR);
        velocity3.copyTo(frame(velocity_rect));

        cv::cvtColor(altitude, altitude3, CV_GRAY2BGR);
        altitude3.copyTo(frame(altitude_rect));

        imshow("JCSAT-14", frame);
        waitKey();
    }


public:

    SpaceXOCR(const std::string& file) 
        : tess(), cap(file)
    {
        if (!cap.isOpened()) {
            throw std::runtime_error("Cannot open file: " + file);
        }        

        fps = cap.get(CAP_PROP_FPS);
        frame_count = cap.get(CAP_PROP_FRAME_COUNT);
        width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
        height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);      

        cout << boost::format("Opened %1% (%2% fps, %3%x%4%)") % file % fps % width % height << endl;

        if (width != 1280 || height != 720) {
            throw std::runtime_error("720p video expected");
        }

        if (tess.Init("D:\\dev\\spacex-ocr\\spacex-telemetry\\spacex-telemetry", "eng")) {
            throw std::runtime_error("Couldn't initialize tesseract");
        }
    }

    Telemetry process() {
        Telemetry result(fps, frame_count);
        
        cv::Rect velocityRect(1041, 142, 72, 20);
        cv::Rect altitudeRect(1181, 142, 72, 20);
        cv::Rect displayRect(541, 142, 144, 40);
        cv::Rect altitudeDisplayRect(541, 342, 144, 40);

        cv::Mat velocity(velocityRect.size() * 2, CV_BGR2GRAY);
        cv::Mat altitude(altitudeRect.size() * 2, CV_BGR2GRAY);

        int prev_digits = 0, prev_velocity = 0;

        cap.set(CAP_PROP_POS_MSEC, 1786 * 1000);// +221000);
        cv::namedWindow("JCSAT-14", WINDOW_AUTOSIZE);
        cv::setMouseCallback("JCSAT-14", onMouse, 0);


        for (int x = 0; ; ++x)
        {
            cap >> frame;
            currentFrame = &frame;

            Size size = frame.size();
            if (frame.empty())
                break;

            configure_for_velocity();
            int ocr_velocity = recognize<int>(frame, velocityRect, velocity);
            int new_digits = count_digits(ocr_velocity);
            if (new_digits < prev_digits) {
                cout << "Missed some characters" << endl;
                int nvelocity = find_component_wise(velocity);
                new_digits = count_digits(nvelocity);

                cout << "Fix " << ocr_velocity << " -> " << nvelocity << endl;
                ocr_velocity = nvelocity;
            }

            configure_for_altitude();
            double ocr_altitude = recognize<double>(frame, altitudeRect, altitude);           

            printf("%.5f, v = %5d, d = %.2f\n", (x / (double)fps), ocr_velocity, ocr_altitude);

            if (prev_velocity > 450 && ocr_velocity < prev_velocity || x == 39) {
                display(velocity, displayRect, altitude, altitudeDisplayRect);
            }

            result.data.emplace_back(ocr_velocity, ocr_altitude * 1000);

            prev_velocity = ocr_velocity;
            prev_digits = new_digits;
            
            if (ocr_velocity == 26720) {
                break;
            }

            /*telemetry.add(ivelocity);

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
            //waitKey(); // waits to display frame*/
        }

        return std::move(result);
    }

};

int main(int argc, char* argv[])
{
    //cv::Mat image = cv::imread("D:\\dev\\spacex-ocr\\spacex-telemetry\\spacex-telemetry\\08893.png");
    //cv::threshold(image, image, 50, 255, cv::THRESH_BINARY);

    std::string file = "E:\\jcsat14.mp4";

    SpaceXOCR ocr(file);
    Telemetry telemetry = ocr.process();

   

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