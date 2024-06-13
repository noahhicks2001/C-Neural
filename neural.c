#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PI 			3.14159265358979323846
#define IMAGE_DATA_OFFSET	16
#define IMAGE_DATA_SIZE		784
#define IMAGE_ROW_SIZE		28
#define IMAGE_COL_SIZE		28

#define LABEL_DATA_OFFSET	8
#define LABEL_DATA_SIZE		1
#define LABEL_VECTOR_SIZE   	10

struct Vector {
	int length;
	double* values;
};

struct Matrix {
	int rows;
	int cols;
	double** values;
};

struct Layer {
	struct Vector* activations;
	struct Vector* z;
	struct Vector* biases;
	struct Matrix* weights;
	struct Matrix* weight_gradients;
	struct Vector* bias_gradients;
	struct Vector* error;
	struct Layer* next;
	struct Layer* prev;
};

struct Model {
	int layer_count;
	struct layer* layers;
	struct Layer* head;
	struct Layer* tail;
};

struct Sample {
	int label_value;
	struct Vector* label_vector;
	struct Vector* image_vector;
};

struct Dataset {
	int size;
	struct Sample* samples;
};

void free_vector(struct Vector* vector);
void free_matrix(struct Matrix* matrix);
void free_dataset(struct Dataset* dataset);
void feed_forward(struct Model* model, struct Sample* sample);
double sigmoid(double x);
double sigmoid_derivative(double x);
int argmax(struct Vector* vector);
int predict(struct Model* model, struct Sample* sample);
void print_accuracy(struct Model* model, struct Dataset* test_dataset);
void zero_vector(struct Vector* vector);
void zero_matrix(struct Matrix* matrix);
void print_vector(struct Vector* vector);
void print_matrix(struct Matrix* matrix);
struct Vector* create_vector(int length);
struct Matrix* create_matrix(int rows, int cols);
void matrix_vector_mult(const struct Matrix* a, const struct Vector* x, struct Vector* b);
void matrix_transpose_vector_mult(const struct Matrix* a, const struct Vector* x, struct Vector* b);
void vector_addition(struct Vector* u, struct Vector* v);
void vector_subtraction(struct Vector* u, struct Vector* v);
void vector_hadamard_product(struct Vector* u, struct Vector* v);
void vector_outer_product(struct Vector* u, struct Vector* v, struct Matrix* a);
void vector_sigmoid_transformation(struct Vector* u, struct Vector* v);
void vector_sigmoid_derivative_transformation(struct Vector* u, struct Vector* v);
double normalize_input_pixel(uint8_t pixel);
void set_dataset_fields(struct Dataset* dataset, int size);
void set_stream_offsets(FILE* image_stream, FILE* label_stream);
void extract_stream_data(struct Dataset* dataset, FILE* image_stream, FILE* label_stream);
void close_streams(FILE* image_stream, FILE* label_stream);
struct Dataset* create_dataset(int size);
void load_mnist(struct Dataset* dataset, char* image_file, char* label_file);
void print_sample(struct Sample* sample);
double normal_distribution();
double uniform_distribution();
void init_parameters(struct Layer* layer);
struct Layer* create_layer(int in_size, int out_size);
struct Model* create_model();
void add_layer(struct Model* model, int in_size, int out_size);
void print_model(struct Model* model);
void zero_neurons(struct Model* model);
void zero_gradients(struct Model* model);
void zero_error(struct Model* model);
void activation(struct Vector* in_features, struct Layer* layer);
void swap(struct Dataset* dataset, int i, int j);
void shuffle_dataset(struct Dataset* dataset);
void feed_forward(struct Model* model, struct Sample* sample);
void compute_output_error(struct Layer* output, struct Sample* sample);
void propagate_error(struct Layer* layer);
void compute_bias_gradients(struct Layer* layer);
void compute_weight_gradients(struct Layer* layer, struct Vector* prev_activations);
void backpropagation(struct Model* model, struct Sample* sample);
void update_weights(struct Layer* layer, double minibatch_size, double learning_rate);
void update_biases(struct Layer* layer, double minibatch_size, double learning_rate);
void apply_sgd_step(struct Model* model, double minibatch_size, double learning_rate);
void SGD(struct Model* model, struct Dataset* minibatch, double learning_rate);
void copy_vector(struct Vector* u, struct Vector* v);
void copy_matrix(struct Matrix* a, struct Matrix* b);
void set_minibatch(struct Dataset* dataset, struct Dataset* minibatch, int index);
void train(struct Model* model, struct Dataset* training_dataset, struct Dataset* test_dataset,
	int epochs, double learning_rate, int minibatch_size);


void free_vector(struct Vector* vector) {
	free(vector->values);
	free(vector);
}

void free_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		free(matrix->values[i]);
	}
	free(matrix);
}

void free_dataset(struct Dataset* dataset) {
	for (int i = 0; i < dataset->size; i++) {
		free_vector(dataset->samples[i].image_vector);
		free_vector(dataset->samples[i].label_vector);
	}
	free(dataset->samples);
	free(dataset);
}


double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
	return sigmoid(x) * (1.0 - sigmoid(x));
}

double normal_distribution() {

	// get uniform rand numbers between 0,1
	double u1 = (double)rand() / (double)RAND_MAX;
	double u2 = (double)rand() / (double)RAND_MAX;

	// use box muller transform
	return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

double uniform_distribution() {
	double random = (double)rand() / (double)RAND_MAX;
	return (random * 2.0) - 1.0;
}

int argmax(struct Vector* vector) {
	int max = 0;
	for (int i = 1; i < vector->length; i++) {
		if (vector->values[i] > vector->values[max]) {
			max = i;
		}
	}
	return max;
}

int predict(struct Model* model, struct Sample* sample) {
	feed_forward(model, sample);
	int prediction = argmax(model->tail->activations);
	return prediction;
}


void print_accuracy(struct Model* model, struct Dataset* test_dataset) {
	int total_correct = 0;
	for (int i = 0; i < test_dataset->size; i++) {
		if (predict(model, &test_dataset->samples[i]) == test_dataset->samples[i].label_value) {
			total_correct++;
		}
	}
	printf("TOTAL CORRECT %i/%i\n", total_correct, test_dataset->size);
}


struct Vector* create_vector(int length) {
	struct Vector* vector = (struct Vector*)malloc(sizeof(struct Vector));
	vector->values = (double*)calloc(length, sizeof(double));
	vector->length = length;
	return vector;
}

struct Matrix* create_matrix(int rows, int cols) {
	struct Matrix* matrix = (struct Matrix*)malloc(sizeof(struct Matrix));
	matrix->values = (double**)malloc(rows * sizeof(double*));
	for (int i = 0; i < rows; i++) {
		matrix->values[i] = (double*)calloc(cols, sizeof(double));
	}
	matrix->rows = rows;
	matrix->cols = cols;
	return matrix;
}

void print_vector(struct Vector* vector) {
	for (int i = 0; i < vector->length; i++) {
		printf("%f \n", vector->values[i]);
	}
}

void print_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->cols; j++) {
			printf("%f ", matrix->values[i][j]);
		}
		printf("\n");
	}
}

void zero_vector(struct Vector* vector) {
	memset(vector->values, 0.0, vector->length * sizeof(double));

}

void zero_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		memset(matrix->values[i], 0.0, matrix->cols * sizeof(double));
	}
}


void copy_vector(struct Vector* u, struct Vector* v) {
	for (int i = 0; i < u->length; i++) {
		u->values[i] = v->values[i];
	}
}

void copy_matrix(struct Matrix* a, struct Matrix* b) {
	for (int i = 0; i < a->rows; i++) {
		for (int j = 0; j < a->cols; j++) {
			a->values[i][j] = b->values[i][j];
		}
	}
}

void matrix_vector_mult(const struct Matrix* a, const struct Vector* x, struct Vector* b) {
	/*
	* computes b <- Ax
	*/
	for (int i = 0; i < a->rows; i++) {
		for (int j = 0; j < a->cols; j++) {
			b->values[i] += a->values[i][j] * x->values[j];
		}
	}
}
void matrix_transpose_vector_mult(const struct Matrix* a, const struct Vector* x, struct Vector* b) {
	/*
	* computes b <-((A)^t)x
	*/
	for (int i = 0; i < a->cols; i++) {
		for (int j = 0; j < a->rows; j++) {
			b->values[i] += a->values[j][i] * x->values[j];
		}
	}
}

void vector_addition(struct Vector* u, struct Vector* v) {
	/*
	* computes u <- u + v
	*/
	for (int i = 0; i < u->length; i++) {
		u->values[i] += v->values[i];
	}
}

void vector_subtraction(struct Vector* u, struct Vector* v) {
	/*
	* computes u <= u - v
	*/
	for (int i = 0; i < u->length; i++) {
		u->values[i] -= v->values[i];
	}
}

void vector_hadamard_product(struct Vector* u, struct Vector* v) {
	/*
	* computes u <- u * v (element wise)
	*/
	for (int i = 0; i < u->length; i++) {
		u->values[i] *= v->values[i];
	}
}

void vector_outer_product(const struct Vector* u, const struct Vector* v, struct Matrix* a) {
	/*
	* computes A <- u(v^T)
	*/
	for (int i = 0; i < u->length; i++) {
		for (int j = 0; j < v->length; j++) {
			a->values[i][j] += u->values[i] * v->values[j];
		}
	}
}

void vector_sigmoid_transformation(struct Vector* u, struct Vector* v) {
	/*
	* computes u <- sigmoid(v) (element wise)
	*/
	for (int i = 0; i < u->length; i++) {
		u->values[i] = sigmoid(v->values[i]);
	}
}

void vector_sigmoid_derivative_transformation(struct Vector* u, struct Vector* v) {
	/*
	* computes u <- sigmoid'(v) (element wise)
	*/
	for (int i = 0; i < u->length; i++) {
		u->values[i] = sigmoid_derivative(v->values[i]);
	}
}


struct Dataset* create_dataset(int size) {
	struct Dataset* dataset = (struct Dataset*)malloc(sizeof(struct Dataset));
	set_dataset_fields(dataset, size);
	return dataset;

}

void set_dataset_fields(struct Dataset* dataset, int size) {
	dataset->samples = (struct Sample*)malloc(sizeof(struct Sample) * size);
	dataset->size = size;
	for (int i = 0; i < dataset->size; i++) {
		dataset->samples[i].image_vector = create_vector(IMAGE_DATA_SIZE);
		dataset->samples[i].label_vector = create_vector(LABEL_VECTOR_SIZE);
		dataset->samples[i].label_value = 0;
	}
}

void print_sample(struct Sample* sample) {
	printf("{%i} \n", sample->label_value);
	for (int i = 0; i < IMAGE_DATA_SIZE; i += IMAGE_COL_SIZE) {
		for (int j = 0; j < IMAGE_ROW_SIZE; j++) {
			if (sample->image_vector->values[i + j] > 0.0) {
				printf("%0.1f ", sample->image_vector->values[i + j]);
			}
			else {
				printf(" ");
			}
		}
		printf("\n");
	}
	printf("\n");
}


void load_mnist(struct Dataset* dataset, char* image_file, char* label_file) {
	FILE* image_stream = fopen(image_file, "rb");
	FILE* label_stream = fopen(label_file, "rb");
	set_stream_offsets(image_stream, label_stream);
	extract_stream_data(dataset, image_stream, label_stream);
	close_streams(image_stream, label_stream);
}


void set_stream_offsets(FILE* image_stream, FILE* label_stream) {
	fseek(image_stream, IMAGE_DATA_OFFSET, SEEK_SET);
	fseek(label_stream, LABEL_DATA_OFFSET, SEEK_SET);
}

double normalize_input_pixel(uint8_t pixel) {
	double value = (double)pixel;
	value /= (double)255;
	return value;
}


 void extract_stream_data(struct Dataset* dataset, FILE* image_stream, FILE* label_stream) {
	uint8_t image_buffer[IMAGE_DATA_SIZE];
	uint8_t label_buffer;
	for (int i = 0; i < dataset->size; i++) {

		// read image/label data to buffers
		fread(image_buffer, sizeof(uint8_t), IMAGE_DATA_SIZE, image_stream);
		fread(&label_buffer, sizeof(uint8_t), LABEL_DATA_SIZE, label_stream);

		// set image data
		for (int j = 0; j < IMAGE_DATA_SIZE; j++) {
			dataset->samples[i].image_vector->values[j] = normalize_input_pixel(image_buffer[j]);
		}

		// set label data
		dataset->samples[i].label_value = (int)label_buffer;
		dataset->samples[i].label_vector->values[(int)label_buffer] = 1.0;
		
	}
}


void close_streams(FILE* image_stream, FILE* label_stream) {
	fclose(image_stream);
	fclose(label_stream);
}


struct Model* create_model() {
	struct Model* model = (struct Model*)malloc(sizeof(struct Model));
	model->layer_count = 0;
	model->layers = NULL;
	model->head = NULL;
	model->tail = NULL;
	return model;
}

void add_layer(struct Model* model, int in_size, int out_size) {
	struct Layer* layer = create_layer(in_size, out_size);
	init_parameters(layer);
	if (model->head == NULL) {
		model->head = layer;
	}
	else if (model->head == model->tail) {
		model->head->next = layer;
		layer->prev = model->head;
	}
	else {
		model->tail->next = layer;
		layer->prev = model->tail;
	}
	model->tail = layer;
	model->layer_count++;
}

void print_model(struct Model* model) {
	printf("LAYER COUNT %i \n", model->layer_count);
	for (struct Layer* layer = model->head; layer != NULL; layer = layer->next) {
		printf("IN %i OUT %i\n", layer->weights->cols, layer->weights->rows);

	}
}

void zero_neurons(struct Model* model) {
	for (struct Layer* layer = model->head; layer != NULL; layer = layer->next) {
		zero_vector(layer->activations);
		zero_vector(layer->z);
	}
}

void zero_gradients(struct Model* model) {
	for (struct Layer* layer = model->head; layer != NULL; layer = layer->next) {
		zero_vector(layer->bias_gradients);
		zero_matrix(layer->weight_gradients);
	}
}

void zero_error(struct Model* model) {
	for (struct Layer* layer = model->head; layer != NULL; layer = layer->next) {
		zero_vector(layer->error);
	}
}

void init_parameters(struct Layer* layer) {
	// init weights
	for (int i = 0; i < layer->weights->rows; i++) {
		for (int j = 0; j < layer->weights->cols; j++) {
			layer->weights->values[i][j] = uniform_distribution();
		}
	}
	// init biases
	for (int i = 0; i < layer->biases->length; i++) {
		layer->biases->values[i] = uniform_distribution();
	}
}

struct Layer* create_layer(int in_size, int out_size) {
	struct Layer* layer = (struct Layer*)malloc(sizeof(struct Layer));
	layer->activations = (struct Vector*)create_vector(out_size);
	layer->z = (struct Vector*)create_vector(out_size);
	layer->biases = (struct Vector*)create_vector(out_size);
	layer->weights = (struct Matrix*)create_matrix(out_size, in_size);
	layer->weight_gradients = (struct Matrix*)create_matrix(out_size, in_size);
	layer->bias_gradients = (struct Vector*)create_vector(out_size);
	layer->error = (struct Vector*)create_vector(out_size);
	layer->next = NULL;
	layer->prev = NULL;
	return layer;
}

void train(struct Model* model, struct Dataset* training_dataset, struct Dataset* test_dataset,
	int epochs,double learning_rate,int minibatch_size) {

	struct Dataset* minibatch = create_dataset(minibatch_size);
	for (int i = 0; i < epochs; i++) {
		printf("EPOCH %i \n", i +1);
		shuffle_dataset(training_dataset);
		for (int j = 0; j < training_dataset->size; j += minibatch_size) {
			set_minibatch(training_dataset, minibatch, j);
			SGD(model, minibatch, learning_rate);
		}
		print_accuracy(model, test_dataset);
	
	}
	free_dataset(minibatch);
}


void shuffle_dataset(struct Dataset* dataset) {
	int rand_index;
	for (int i = dataset->size - 1; i > 0; i--) {
		rand_index = (int)rand() % (i + 1);
		swap(dataset, i, rand_index);
	}
}


void copy_sample(struct Sample* a, struct Sample* b) {
	/*
	* copies from a <- b
	*/
	copy_vector(a->image_vector, b->image_vector);
	copy_vector(a->label_vector, b->label_vector);
	a->label_value = b->label_value;

}
void swap(struct Dataset* dataset, int i, int j) {

	// create temporary sample to swap structs (deep copy vs shallow)
	struct Sample* temp = (struct Sample*)malloc(sizeof(struct Sample));
	temp->image_vector = create_vector(IMAGE_DATA_SIZE);
	temp->label_vector = create_vector(LABEL_VECTOR_SIZE);

	// store sample
	copy_sample(temp, &dataset->samples[i]);
	
	// swap sample by index
	copy_sample(&dataset->samples[i], &dataset->samples[i]);

	// copy stored sample
	copy_sample(&dataset->samples[j], temp);

	// free temp sample
	free_vector(temp->image_vector);
	free_vector(temp->label_vector);
	free(temp);

}

void set_minibatch(struct Dataset* dataset, struct Dataset* minibatch, int index) {
	for (int i = 0; i < minibatch->size; i++) {
		copy_vector(minibatch->samples[i].image_vector, dataset->samples[index + i].image_vector);
		copy_vector(minibatch->samples[i].label_vector, dataset->samples[index + i].label_vector);
		minibatch->samples[i].label_value = dataset->samples[index + i].label_value;
	}
}

void SGD(struct Model* model, struct Dataset* minibatch, double learning_rate) {
	for (int i = 0; i < minibatch->size; i++) {
		feed_forward(model, &minibatch->samples[i]);		// feed sample into network
		backpropagation(model, &minibatch->samples[i]);
	}
	apply_sgd_step(model, (double)minibatch->size, learning_rate);
	zero_gradients(model);
}


void feed_forward(struct Model* model, struct Sample* sample) {
	struct Layer* layer = model->head;
	zero_neurons(model);						// clear neurons before feeding in data
	activation(sample->image_vector, layer);			// feed input data
	for (; layer->next != NULL; layer = layer->next) {
		activation(layer->activations, layer->next);		// compute next layers activation
	}
}

void activation(struct Vector* in_features, struct Layer* layer) {
	// compute weighted sum
	matrix_vector_mult(layer->weights, in_features, layer->z);

	// add biases
	vector_addition(layer->z, layer->biases);

	// apply sigmoid
	vector_sigmoid_transformation(layer->activations, layer->z);
}

void backpropagation(struct Model* model, struct Sample* sample) {
	// compute inital outout error
	struct Layer* layer = model->tail;
	compute_output_error(layer, sample);
	compute_bias_gradients(layer);
	compute_weight_gradients(layer, layer->prev->activations);

	for (layer = layer->prev; layer != NULL; layer = layer->prev) {
		propagate_error(layer);
		if (layer == model->head) {
			compute_weight_gradients(layer, sample->image_vector);
		}
		else {
			compute_weight_gradients(layer, layer->prev->activations);
		}
		compute_bias_gradients(layer);
	}
	zero_error(model);
}

void compute_output_error(struct Layer* output, struct Sample* sample) {
	// compute (a(l) - y)
	for (int i = 0; i < output->error->length; i++) {
		output->error->values[i] = (output->activations->values[i] -
			sample->label_vector->values[i]);
	}
}

void propagate_error(struct Layer* layer) {
	// compute s <- ((wl+1)^T) sl+1
	matrix_transpose_vector_mult(layer->next->weights, layer->next->error, layer->error);

	// compute z <- sigmoid'(z)
	vector_sigmoid_derivative_transformation(layer->z, layer->z);

	// compute s <- s * sigmoid'(z)
	vector_hadamard_product(layer->error, layer->z);
}
void compute_bias_gradients(struct Layer* layer) {
	// add error to bias gradient total
	vector_addition(layer->bias_gradients, layer->error);
}

void compute_weight_gradients(struct Layer* layer, struct Vector* prev_activations) {
	// compute weight total with Wg <- s(a-1^T)
	vector_outer_product(layer->error, prev_activations, layer->weight_gradients);
}


void apply_sgd_step(struct Model* model, double minibatch_size, double learning_rate) {
	for (struct Layer* layer = model->head; layer != NULL; layer = layer->next) {
		update_weights(layer, minibatch_size, learning_rate);
		update_biases(layer, minibatch_size, learning_rate);
	}
}

void update_weights(struct Layer* layer, double minibatch_size, double learning_rate) {
	for (int i = 0; i < layer->weights->rows; i++) {
		for (int j = 0; j < layer->weights->cols; j++) {
			layer->weights->values[i][j] -= (((learning_rate / minibatch_size) *
				layer->weight_gradients->values[i][j]));
		}

	}
}

void update_biases(struct Layer* layer, double minibatch_size, double learning_rate) {
	for (int i = 0; i < layer->biases->length; i++) {
		layer->biases->values[i] -= ((learning_rate / minibatch_size) *
			(layer->bias_gradients->values[i]));

	}
}


int main()  {

	// set random seed for parameter init
	srand(time(NULL));

	struct Dataset* training_dataset = create_dataset(60000);
	struct Dataset* test_dataset = create_dataset(10000);

	load_mnist(training_dataset, "train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	load_mnist(test_dataset, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	

	struct Model* model = create_model();
	add_layer(model, 784, 100);
	add_layer(model, 100,10);
	train(model, training_dataset, test_dataset, 1, 3.0, 10);
	

	for (int i = 0; i < 10; i++) {
		print_sample(&test_dataset->samples[i]);
		printf("PREDICTION %i \n\n", predict(model, &test_dataset->samples[i]));
	}
	

	free_dataset(training_dataset);
	free_dataset(test_dataset);
	

}

