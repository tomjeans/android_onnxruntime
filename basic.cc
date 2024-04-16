#include <onnxruntime_cxx_api.h>
#include <nnapi_provider_factory.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

//xtensor
#include <math.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xvectorize.hpp>
#include <algorithm>

template<typename T>
void show_vector(std::vector <T> &value)
{
    std::cout << "===the vector values ===" << std::endl;

    for (auto single_value : value)
    {
        std::cout << " " << single_value << " ";
    }

    std::cout << "===the vector values ===" << std::endl;
}






float iou_real(int box_a_topx, int box_a_topy, int box_a_w, int box_a_h, int box_b_topx, int box_b_topy, int box_b_w, int box_b_h)
{
    const float area_a = box_a_w * box_a_h;
    const float area_b = box_b_w * box_b_h;

    // 计算重叠区域的坐标范围
    const float x1 = std::max(box_a_topx, box_b_topx);
    const float y1 = std::max(box_a_topy, box_b_topy);
    const float x2 = std::min(box_a_topx + box_a_w, box_b_topx + box_b_w);
    const float y2 = std::min(box_a_topy + box_a_h, box_b_topy + box_b_h);

    // 计算重叠区域的面积
    const float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    // 计算并集区域的面积
    const float union_area = area_a + area_b - intersection_area;

    // 计算IoU
    return union_area > 0 ? intersection_area / union_area : 0;
}


xt::xtensor<int, 2> nms_simple(xt::xtensor<int, 2> boxes_int, std::vector<double> scores_filter, float threshold)
{
    std::map<double, xt::xarray<int>, std::greater<double>> alls;

    



    for (int inds = 0; inds < boxes_int.shape(0); inds++)
    {
        
        xt::xarray<int> single_box = xt::view(boxes_int, inds, xt::all());
        
        alls.insert(std::pair<double, xt::xarray<int>>(scores_filter[inds], single_box));
    }

  



    std::vector<xt::xarray<int>> results_boxs_temp;
    std::vector<double> scores_temp;

    



    for (auto single : alls)
    {
        scores_temp.push_back(single.first);
        results_boxs_temp.push_back(single.second);
    }
   




    std::vector<xt::xarray<int>> results_boxs_final;
    std::vector<double> scores_final;

   



    for (int i = 0; i < results_boxs_temp.size(); ++i)
    {
        bool keep = true;
        for (int j = 0; j < results_boxs_final.size(); ++j)
        {

            if (iou_real(results_boxs_temp[i](0), results_boxs_temp[i](1), results_boxs_temp[i](2), results_boxs_temp[i](3), results_boxs_final[j](0), results_boxs_final[j](1), results_boxs_final[j](2), results_boxs_final[j](3)) > threshold)
            {
                keep = false;
                break;

            }
        }
        if (keep)
        {
            results_boxs_final.push_back(results_boxs_temp[i]);
            scores_final.push_back(scores_temp[i]);
        }
    }
    xt::xtensor<double, 2> result_z = xt::zeros<double>({ (int)results_boxs_final.size(),5 });
    



    for (int index_boxs = 0; index_boxs < results_boxs_final.size(); index_boxs++)
    {
        result_z(index_boxs, 0) = (double)results_boxs_final[index_boxs](0)- (double)results_boxs_final[index_boxs](2)/2.0;
        result_z(index_boxs, 1) = (double)results_boxs_final[index_boxs](1)- (double)results_boxs_final[index_boxs](3)/2.0;
        result_z(index_boxs, 2) = (double)results_boxs_final[index_boxs](2);
        result_z(index_boxs, 3) = (double)results_boxs_final[index_boxs](3);
        result_z(index_boxs, 4) = scores_final[index_boxs];
    }

    std::cout << "run in there 8" << std::endl;

    std::cout << "the last results" << std::endl;
    std::cout << result_z << std::endl;
    std::cout << "the last results" << std::endl;


    //
    //y = np.copy(x)
    //y[..., 0] = x[..., 0] - x[..., 2] / 2
    //y[..., 1] = x[..., 1] - x[..., 3] / 2
    //y[..., 2] = x[..., 0] + x[..., 2] / 2
    //y[..., 3] = x[..., 1] + x[..., 3] / 2





    return result_z;



}


using namespace std;
using namespace Ort;

cv::Mat resizeim(cv::Mat img,int w,int h)
{

    cv::Mat image_rgb=img.clone();
    cv::Mat resized;
    cv::resize(image_rgb, resized, cv::Size(w, h));
    return resized;
    
}



std::vector<float> normalize_(cv::Mat img)
{
    std::vector<float> input_image_;
    //    img.convertTo(img, CV_32F);
    int row = img.rows;
    int col = img.cols;
    input_image_.resize(row * col * img.channels());
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];       // HWC to CHW, BGR to RGB，j * 3 + 2 - c
                input_image_[c * row * col + i * col + j] = pix / 255.0;
            }
        }
    }
    return input_image_;
}





int main()
{
// Load the model and create InferenceSession
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "NNAPI");
    std::string model_path = "yolov9-c.onnx";
    //wchar_t* wc = new wchar_t[model_path.size()];
    //swprintf(wc, 100, L"%S", model_path.c_str());

	//origin_data
    //Ort::Session session(env, model_path.c_str(), Ort::SessionOptions{nullptr});
	
	//newapi

	
	
	
	Ort::SessionOptions sessionOp;

	sessionOp.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	std::cout<<"run there"<<std::endl;
	uint32_t nnapi_flag = 0;
    nnapi_flag |= NNAPI_FLAG_CPU_DISABLED;
    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOp, nnapi_flag));
	std::cout<<"run there 2"<<std::endl;
	Ort::Session session(env,model_path.c_str(),sessionOp);
	
	
	std::cout<<"run there 3"<<std::endl;
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
	std::cout<<"run there 4"<<std::endl;

    std::cout << "num_input_nodes :" << num_input_nodes << std::endl;
    std::cout << "num_output_nodes :" << num_output_nodes << std::endl;

	std::vector<const char* > input_names;
    std::vector<const char* > output_names;
    std::vector<vector<int64_t>> input_node_dims; // >=1 outputs
    std::vector<vector<int64_t>> output_node_dims; // >=1 outputs
    std::vector<AllocatedStringPtr> In_AllocatedStringPtr;
    std::vector<AllocatedStringPtr> Out_AllocatedStringPtr;


    std::vector<int> inputshape;
    std::vector<int> outputshape;

    //set inputs params
    for (int i = 0; i < num_input_nodes; i++)                         // onnxruntime1.12版本后不能按照从前格式写
    {
        AllocatorWithDefaultOptions allocator;                              // 配置输入输出节点内存
        In_AllocatedStringPtr.push_back(session.GetInputNameAllocated(i, allocator));
        input_names.push_back(In_AllocatedStringPtr.at(i).get());           // 内存
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);   // 类型
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();                     // 输入shape
        input_node_dims.push_back(input_dims);                              // 输入维度信息

        //
        for (auto ssi : input_node_dims[0])
        {
            std::cout << "the input_dims :" << ssi << std::endl;
            inputshape.push_back(int(ssi));
        }
        //


    }

    for (int i = 0; i < num_output_nodes; i++)
    {
        AllocatorWithDefaultOptions allocator;
        Out_AllocatedStringPtr.push_back(session.GetOutputNameAllocated(i, allocator));
        output_names.push_back(Out_AllocatedStringPtr.at(i).get());
        std::cout << "output_names :" << output_names[0] << std::endl;
        Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
        //
        for (auto ss : output_node_dims[0])
        {
            std::cout << "the output_dims :" << ss << std::endl;
            outputshape.push_back(int(ss));
        }
        //
    }




    cv::Mat imags = cv::imread("test.jpg");

    cv::Mat res_img=resizeim(imags, inputshape[2], inputshape[3]);

    std::vector<float> input_image_ =normalize_(res_img);

    array<int64_t, 4> input_shape_{ 1, 3, inputshape[2], inputshape[3] };

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	
	auto start_time = chrono::steady_clock::now();
	for(int times=0;times<10;times++)
	{
    std::vector<Ort::Value> ort_outputs = session.Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());


    std::cout << "\n";
    std::cout << "\n";
    std::cout << "\n";

    std::cout << " the ort_outputs shape :" << ort_outputs.size() << std::endl;

    float* all_data = ort_outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> _outputTensorShape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();



    std::cout << "_outputTensorShape [0] :" << _outputTensorShape[0] << " " << _outputTensorShape[1] <<" "<<_outputTensorShape[2]<<" "<< std::endl;


    int dim1 = _outputTensorShape[0];
    int dim2 = _outputTensorShape[1];
    int dim3 = _outputTensorShape[2];
	
	auto end_time = chrono::steady_clock::now();
	
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	std::cout<<"runtime :"<<duration/10.0<<" ms"<<std::endl;
    std::vector<float> results(dim1 * dim2 * dim3);
	
	memcpy(&results[0], all_data, sizeof(float) * dim1 * dim2 * dim3);
	
	
	xt::xtensor<double, 3>::shape_type shape_x = { static_cast<unsigned long>(dim1),static_cast<unsigned long>(dim2),static_cast<unsigned long>(dim3)};
    xt::xtensor<double, 3> tn1 = xt::adapt(results, shape_x);

    xt::xtensor<double, 2> prediction = xt::view(tn1, 0, xt::all(), xt::all());
    std::cout << prediction << std::endl;

    xt::xtensor<double, 2> prediction_t = xt::transpose(prediction, { 1,0 });


    auto scores_t = xt::view(prediction_t, xt::all(), xt::range(4, _));

    xt::xarray <int> index = xt::argmax(scores_t, 1);




    std::vector <double> scores;

    for (int index_t=0;index_t<scores_t.shape(0);index_t++)
    {

        scores.push_back(scores_t(index_t, index(index_t)));
    }
   
    float con_thresold = 0.4;


    std::vector<std::size_t> shape = { 1, index.shape(0) };
    xt::xarray<double> a1 = xt::adapt(scores, shape);


    std::vector<double> scores_filter;

    std::vector<int> index_s;
    for (int ins = 0; ins < scores.size(); ins++)
    {
        if (scores[ins] > con_thresold)
        {
            index_s.push_back(ins);
            scores_filter.push_back(scores[ins]);
            
        }
    }



    xt::xtensor<double, 2> prediction_zero = xt::zeros<double>({ index_s.size(),prediction_t.shape(1) });



    std::cout << index_s.size() << " " << prediction.shape(1) << std::endl;


    for (int index_p = 0; index_p < prediction_zero.shape(0); index_p++)
    {
        for (int index_p_t = 0; index_p_t < prediction_zero.shape(1); index_p_t++)
        {
            
            prediction_zero(index_p, index_p_t) = prediction_t(index_s[index_p], index_p_t);
        }
    }



    xt::xtensor<double, 2> prediction_zero_filter = xt::view(prediction_zero, xt::all(), xt::range(4, _));



    xt::xarray <int> class_id = xt::argmax(prediction_zero_filter, 1);



    xt::xtensor<double, 2> boxes = xt::view(prediction_zero, xt::all(), xt::range(_, 4));



    int width = inputshape[2];
    int height = inputshape[3];
    std::vector<int> input_shape = { width,height };


    int imag_width = imags.cols;
    int imag_height = imags.rows;
    std::vector<int> img_shape = { imag_width,imag_height };

    int rows_ = boxes.shape(0);
    int cols_ = boxes.shape(1);


    for (int index_j = 0; index_j < rows_; index_j++)
    {
        boxes(index_j, 0) = boxes(index_j, 0) / input_shape[0];
        boxes(index_j, 1) = boxes(index_j, 1) / input_shape[1];
        boxes(index_j, 2) = boxes(index_j, 2) / input_shape[0];
        boxes(index_j, 3) = boxes(index_j, 3) / input_shape[1];
    }


    for (int index_n = 0; index_n < rows_; index_n++)
    {
        boxes(index_n, 0) = boxes(index_n, 0)*img_shape[0];
        boxes(index_n, 1) = boxes(index_n, 1)*img_shape[1];
        boxes(index_n, 2) = boxes(index_n, 2)*img_shape[0];
        boxes(index_n, 3) = boxes(index_n, 3)*img_shape[1];
    }

    

    xt::xtensor<int, 2> boxes_int = xt::cast<int>(boxes);
    





    std::cout << "the scores" << std::endl;
    for (auto single_score : scores_filter)
    {
        std::cout << single_score << " ";
    }
    std::cout << " the scores " << std::endl;


    xt::xtensor<int, 2> res = nms_simple(boxes_int, scores_filter, 0.4);



    for (int box_i = 0; box_i < res.shape(0); box_i++)
    {
        cv::Rect rec1;
        rec1.x = res(box_i, 0);
        rec1.y = res(box_i, 1);
        rec1.width = res(box_i, 2);
        rec1.height = res(box_i, 3);
        std::cout << "the class " << class_id[box_i] << std::endl;
        
        cv::rectangle(imags, rec1, cv::Scalar(0,255,0), 1, 1, 0);
        
    }
    cv::imwrite("result.jpg",imags);
	}
	auto end_time = chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> elapsed = end_time-start_time;
	std::cout<<"duration time :"<<elapsed.count()<<std::endl;
}