#define NOMINMAX

#include <opencv2/text.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <array>
#include <random>
#include <vector>
#include <numeric>

using std::cout;
using std::cerr;
using std::endl;

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

    void write_data(std::ostream& out) {
        for (size_t x = 0; x < data.size(); x++) {
            const TelemetryPoint& point = data[x];
            out << x/(double)fps << "," << point.velocity << "," << point.altitude << endl;
        }
    }
};

void onMouse(int event, int x, int y, int, void* userdata)
{
    if (event != CV_EVENT_LBUTTONDOWN)
        return;

    cv::Mat* frame = static_cast<cv::Mat*>(userdata);

    cout << "(" << x << ", " << y << "): " << (int)frame->at<uchar>(y, x) << endl;
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

class Trainer {
private:
    std::array<int, 10> counts;
    std::vector<cv::Mat> training_data;
    int width, height, count, total_count;

public:
    Trainer(int width, int height, int count)
        : width(width), height(height), count(count), total_count(count*10)          
    {
        std::fill(counts.begin(), counts.end(), 0);        
        counts[5] -= 3;
        counts[6] -= 3;
    }

    void seen_digit(int digit, const cv::Mat& mat)
    {
        if (counts[digit] < count) {
            cv::Mat copy;
            mat.copyTo(copy);           

            training_data.push_back(std::move(copy));
            counts[digit]++;
        }
    }

    bool is_ready() const {
        return training_data.size() == total_count;
    }

    cv::Mat generate_training_image() const {

        int vertical_padding = 10;
        int horizontal_padding = 0;

        cv::Mat training_mat(cv::Size(width + (horizontal_padding*2), (height + vertical_padding) * total_count), CV_8UC1);
        training_mat.setTo(cv::Scalar(0));

        std::vector<int> order(training_data.size());
        std::iota(order.begin(), order.end(), 0);      
        std::random_shuffle(order.begin(), order.end());

        cv::Rect next_place(horizontal_padding, vertical_padding, width, height);
        for (size_t x = 0; x < order.size(); x++) {
            const cv::Mat& mat = training_data[order[x]];
            mat.copyTo(training_mat(next_place));

            next_place.y += height + vertical_padding;
        }

        return training_mat;
    }
};

struct MinimumInt {
    int x;

    MinimumInt(int x) 
        : x(x)
    {}

    bool operator==(const MinimumInt& other) {
        return other.x >= x;
    }
};

class SpaceXOCR {
private:
    tesseract::TessBaseAPI tess;
    cv::VideoCapture cap;
    cv::Mat frame;
    cv::Rect velocity_rect;
    cv::Rect altitude_rect;
    cv::Mat velocity_mat, altitude_mat;   
    Trainer trainer;

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

    bool get_velocity(const cv::Rect& roi, cv::Mat& mat, char* result)
    {
        resize(frame(roi), mat, roi.size() * 2, 0.0, 0.0, cv::INTER_CUBIC);
        cv::cvtColor(mat, mat, CV_BGR2GRAY);

        // accuracy is very dependent on this threshold value
        cv::threshold(mat, mat, 130, 255, cv::THRESH_BINARY);

        tess.SetImage((uchar*)mat.data, mat.size().width, mat.size().height, mat.channels(), mat.step1());

        int boxes[] = { 7, 33, 60, 89, 114 };

        for (int i = 0; i < 5; i++) {
            int x = boxes[i] - 1;
            int y = 3;
            int w = 23 + 1;
            int h = 32;

            tess.SetRectangle(x, y, w, h);
            char c = tess.GetUTF8Text()[0];

            if (c < '0' || c > '9') {
                return false;
            }
            result[i] = c;
            int int_value = c - '0';

            //int conf = tess.MeanTextConf();
            //fprintf(stdout, "Box[%d]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %d\n",
            //        i, x, y, w, h, conf, int_value);            
        }

        result[5] = 0;
        return true;
        //for (int i = 0; i < 5; i++) {
        //    cv::line(mat, cv::Point(boxes[i]-1, 0), cv::Point(boxes[i]-1, 36), cv::Scalar(255, 255, 255));
        //}
    }

    void display(cv::Mat& velocity, cv::Rect& velocity_rect, cv::Mat& altitude, cv::Rect& altitude_rect) {
        cv::Mat velocity3, altitude3;

        cv::cvtColor(velocity, velocity3, CV_GRAY2BGR);
        velocity3.copyTo(frame(velocity_rect));

        cv::cvtColor(altitude, altitude3, CV_GRAY2BGR);
        altitude3.copyTo(frame(altitude_rect));

        imshow("JCSAT-14", frame);
        cv::waitKey();
    }

    bool skip_to_approx_telemetry_start(int frame_step)
    {
        unsigned long current_frame = cap.get(cv::CAP_PROP_POS_FRAMES);
        char velocity_str[6];

        cap >> frame;

        while (!frame.empty()) {       
            imshow("JCSAT-14", frame);
            cv::waitKey(1);

            if (get_velocity(velocity_rect, velocity_mat, velocity_str)) {
                cap.set(cv::CAP_PROP_POS_FRAMES, std::max(0ul, current_frame - frame_step));
                return true;
            }

            current_frame += frame_step;
            cap.set(cv::CAP_PROP_POS_FRAMES, current_frame);

            cap >> frame;
        }

        return false;
    }

    template <typename Itr>
    bool skip_to_telemetry_start(const Itr expected_start, const Itr expected_end) {
        int frame_step = roundl(fps * 30);
        char velocity_str[6];

        if (skip_to_approx_telemetry_start(frame_step)) {
            auto itr = expected_start;
            while (itr != expected_end) {

                cap >> frame;

                if (frame.empty()) {
                    return false;
                }

                if (!get_velocity(velocity_rect, velocity_mat, velocity_str)) {
                    continue;
                }

                int velocity = atoi(velocity_str);

                if (*itr == velocity) {
                    ++itr;
                } else {
                    itr = expected_start;
                }
            }

            return true;
        } else {
            return false;
        }
    }

    void begin_processing_telemetry(Telemetry& result)
    {
        cv::Rect display_rect(541, 142, 144, 40);
        cv::Rect altitude_display_rect(541, 342, 144, 40);

        int prev_digits = 0;
        int prev_velocity = 0;

        bool do_show = false;
        int empty_count = 0;

        for (int x = 0; ; ++x)
        {
            cap >> frame;

            cv::Size size = frame.size();
            if (frame.empty())
                break;

            configure_for_velocity();

            char velocity_str[6];
            int boxes[] = { 7, 33, 60, 89, 114 };
            if (!get_velocity(velocity_rect, velocity_mat, velocity_str)) {

                ++empty_count;
                //imshow("JCSAT-14", frame);
                //cv::waitKey();

                if (empty_count >= 5) {
                    unsigned long current = cap.get(cv::CAP_PROP_POS_FRAMES);
                    std::array<MinimumInt, 3> expected_mins{ prev_velocity - 1000, prev_velocity - 1000, prev_velocity - 1000 };

                    if (skip_to_telemetry_start(expected_mins.begin(), expected_mins.end())) {
                        unsigned long skipped = cap.get(cv::CAP_PROP_POS_FRAMES) - current;
                        for (unsigned long i = 0; i < skipped; i++) {
                            result.data.emplace_back(0, 0);
                        }

                        empty_count = 0;
                    } else {
                        break;
                    }                    
                }
            } else {
                empty_count = 0;
            }

           /* for (int i = 0; i < 5; i++) {
                int x1 = boxes[i] - 1;
                int y1 = 3;
                int w = 23 + 1;
                int h = 32;

                trainer.seen_digit(velocity_str[i] - '0', velocity_mat(cv::Rect(x1, y1, w, h)));

                if (trainer.is_ready()) {
                    cv::imwrite("training1.tif", trainer.generate_training_image());
                    throw std::runtime_error("finish");
                    //imshow("JCSAT-14", trainer.generate_training_image());
                    //cv::waitKey();
                }
            }*/

            int ocr_velocity = atoi(velocity_str);
            //if (ocr_velocity >= 9177) {
            //    cout << "Break" << endl;
            //    do_show = true;
            //}
        
            configure_for_altitude();
            double ocr_altitude = recognize<double>(frame, altitude_rect, altitude_mat);

            printf("%.5f, v = %5d, d = %.2f\n", (x / (double)fps), ocr_velocity, ocr_altitude);

            if (ocr_velocity < prev_velocity && (ocr_velocity > 8135 || ocr_velocity < 8210)) {
               // display(velocity_mat, display_rect, altitude_mat, altitude_display_rect);
            }

            if (do_show) {
                display(velocity_mat, display_rect, altitude_mat, altitude_display_rect);
            }

            //find_component_wise(velocity_mat);
            //display(velocity_mat, display_rect, altitude_mat, altitude_display_rect);
            
            result.data.emplace_back(ocr_velocity, ocr_altitude * 1000);

            prev_velocity = ocr_velocity;

            //if (ocr_velocity >=27106) { // crs8
            //if (ocr_velocity >= 26980) { // ses9
            //if (ocr_velocity >= 26720) { // jcsat14
            //    break;
            //}
        }
    }

public:

    SpaceXOCR(const std::string& file) 
        : tess(), cap(file), velocity_rect(1041, 142, 72, 20), altitude_rect(1178, 140, 72, 20),
          velocity_mat(velocity_rect.size() * 2, CV_BGR2GRAY), altitude_mat(altitude_rect.size() * 2, CV_BGR2GRAY),
          trainer(24, 32, 10)
    {
        if (!cap.isOpened()) {
            throw std::runtime_error("Cannot open file: " + file);
        }        

        fps = cap.get(cv::CAP_PROP_FPS);
        frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
        width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        cout << boost::format("Opened %1% (%2% fps, %3%x%4%)") % file % fps % width % height << endl;

        if (width != 1280 || height != 720) {
            throw std::runtime_error("720p video expected");
        }

        //if (tess.Init("C:\\Program Files (x86)\\Tesseract-OCR", "spacex")) {
        if (tess.Init("D:\\dev\\spacex-ocr\\spacex-telemetry\\spacex-telemetry", "spacex")) {
            throw std::runtime_error("Couldn't initialize tesseract");
        }
    }

    Telemetry process() {
        Telemetry result(fps, frame_count);       
        
        //cap.set(cv::CAP_PROP_POS_MSEC, 603 * 1000 + 2000);// +221000);
        //cap.set(cv::CAP_PROP_POS_MSEC, 1786 * 1000);
        cv::namedWindow("JCSAT-14", cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback("JCSAT-14", onMouse, &frame);

        cout << "Searching for liftoff.." << endl;

        std::array<int, 2> start_seq{ 0, 1 };
        if (!skip_to_telemetry_start(start_seq.begin(), start_seq.end())) {
            throw std::runtime_error("Couldn't locate start of telemetry");
        }        

        begin_processing_telemetry(result);
        return std::move(result);
        cout << "Found approximate liftoff time.. Searching for T-0..." << endl;

        if (tess.Init("D:\\dev\\spacex-ocr\\spacex-telemetry\\spacex-telemetry", "spacex")) {
            throw std::runtime_error("Couldn't initialize tesseract for digits");
        }

        // Get our first reading..
        configure_for_velocity();
        int prev_velocity = 0;

        for (int x = 0; x < fps * 20; x++) {
            cap >> frame;

            if (frame.empty()) {
                break;
            }

            
            char velocity_str[6];
            if (!get_velocity(velocity_rect, velocity_mat, velocity_str))// recognize<int>(frame, velocity_rect, velocity_mat);
                continue;
            int ocr_velocity = atoi(velocity_str);
            cout << ocr_velocity << endl;

            if (prev_velocity == 0 && ocr_velocity == 1) {
                double liftoff = cap.get(cv::CAP_PROP_POS_MSEC) / 1000;
                printf("Liftoff at %.2f seconds into video\n", liftoff);

                imshow("JCSAT-14", frame);
                cv::waitKey();

                begin_processing_telemetry(result);
                return std::move(result);
            }

            prev_velocity = ocr_velocity;
        }   

        throw std::runtime_error("Couldn't find precise liftoff frame");
    }

};

int main(int argc, char* argv[])
{
    try {
        //SpaceXOCR ocr("D:\\dev\\spacex-ocr\\ses9.mp4");
        //SpaceXOCR ocr("D:\\dev\\spacex-ocr\\crs8.mp4");
        SpaceXOCR ocr("E:\\jcsat14.mp4");
        Telemetry telemetry = ocr.process();

        std::ofstream myfile;
        myfile.open("velocity.csv");
        telemetry.write_data(myfile);
    } catch (std::exception& e) {
        cerr << "Failed: " << e.what() << endl;
    }

     return 0;
}