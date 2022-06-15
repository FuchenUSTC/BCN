#ifndef CAFFE_SOFTMAX_BI_CALIBRATE_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAX_BI_CALIBRATE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

	/**
	* @brief Computes the softmax loss with Bi-Calibration (including selection scheme)
	*
	* SoftmaxLayer + Bi-Calibration (t2q and q2t, how to select between them)
	* At test time, this layer can be replaced simply by a SoftmaxLayer.
	*
	*
	*/

	template <typename Dtype>
	class SoftmaxBiCalibrateLossLayer : public LossLayer<Dtype> {
	public:
		/**
		* @param param provides LossParameter loss_param, with options:
		*  - ignore_label (optional)
		*    Specify a label value that should be ignored when computing the loss.
		*  - normalize (optional, default true)
		*    If true, the loss is normalized by the number of (nonignored) labels
		*    present; otherwise the loss is simply summed over spatial locations.
		*/

		explicit SoftmaxBiCalibrateLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "SoftmaxBiCalibrateLoss"; }
		virtual inline int ExactNumTopBlobs() const { return -1; }
		virtual inline int ExactNumBottomBlobs() const { return 4; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		/**
		* @brief Computes the softmax loss error gradient w.r.t. the predictions.
		*
		* Gradients cannot be computed with respect to the label inputs (bottom[1] and bottom[3]),
		* so this method ignores bottom[1] and bottom[3] and requires !propagate_down[1] && !propagate_down[3], crashing
		* if propagate_down[1] and propagate_down[3] is set.
		*
		*/

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		/// Read the normalization mode parameter and compute the normalizer based
		/// on the blob size.  If normalization_mode is VALID, the count of valid
		/// outputs will be read from valid_count, unless it is -1 in which case
		/// all outputs are assumed to be valid.
		virtual Dtype get_normalizer(
			LossParameter_NormalizationMode normalization_mode, int valid_count);

		/// compute the text-to-query correction by bottom-up aggregation
		virtual void get_t2q_correction(
			const Dtype* text_prob_data, int query_dim);

		/// compute the query-to-text correction by top-down decomposition
		virtual void get_q2t_correction(
			const Dtype* query_prob_data, int text_dim);

		/// compute the l2 norm 
		virtual Dtype compute_l2_norm(vector<Dtype> vector_a, vector<Dtype> vector_b);

		/// obtatin the refined query and text supervision 
		/// through Calibration Selection Scheme
		virtual void get_refined_query_text_label(
			const Dtype* query_prob_data, int query_label, const Dtype* text_prob_data,
			const Dtype* text_label, int query_dim, int text_dim);

		/// compute the differences between correction and confidence score
		/// the differnces are utilized for Calibration Selection
		virtual void get_differences(
			const Dtype* query_prob_data, int query_label, const Dtype* text_prob_data,
			const Dtype* text_label, int query_dim, int text_dim);

		/// load the label mapping between query and text prototype
		virtual void build_query_text_map();


		/// store the corresponding probability
		std::multimap<int, int> query_text_cluster_map; // the query and text prototype mapping, one query -- multiple text prototype
		std::map<int, int> text_cluster_query_map;      // the text prototype and query mapping, one text prototype -- one query
		Dtype q_threshold; // $\varepsilon^q$ for calibration selection
		Dtype t_threshold; // $\varepsilon^t$ for calibration selection
		Dtype alpha;       // $\alpha$ for moving average
		Dtype t2q_diff;    // l2 norm of the differnce bettwen t2q correction and the query confidence
		Dtype q2t_diff;    // l2 norm of the differnce between q2t correction and the text confidence 
		vector<Dtype> refined_query_label;             // temp refined query supervision
		vector<vector<Dtype>> refined_query_label_all; // all the refined query supervision in a batch
		vector<Dtype> refined_text_label;              // temp refined text supervision
		vector<vector<Dtype>> refined_text_label_all;  // all the refined text supervision in a batch
		vector<Dtype> t2q_correction_vec_;             // temp t2q correction through bottom-up aggregation
		vector<Dtype> q2t_correction_vec_;             // temp q2t correction through top-down decomposition
		vector<vector<Dtype>> history_text_label_all;  // array to store the history text supervision

		/// The internal SoftmaxLayer used to map predictions to a distribution.
		shared_ptr<Layer<Dtype> > softmax_layer_;
		shared_ptr<Layer<Dtype> > softmax_text_layer_;
		/// prob stores the output probability predictions from the SoftmaxLayer.
		Blob<Dtype> prob_;
		Blob<Dtype> prob_text_;
		/// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
		vector<Blob<Dtype>*> softmax_bottom_vec_;
		vector<Blob<Dtype>*> softmax_bottom_text_vec_;
		/// top vector holder used in call to the underlying SoftmaxLayer::Forward
		vector<Blob<Dtype>*> softmax_top_vec_;
		vector<Blob<Dtype>*> softmax_top_text_vec_;
		/// Whether to ignore instances with a certain label.
		bool has_ignore_label_;
		/// The label indicating that an instance should be ignored.
		int ignore_label_;
		/// How to normalize the output loss.
		LossParameter_NormalizationMode normalization_;

		int softmax_axis_, outer_num_, inner_num_, train_index_, num_train_, epoch_index_;
	};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_BI_CALIBRATE_LOSS_LAYER_HPP_
