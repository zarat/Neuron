double sigmoid(double x);
double relu(double x);
double leaky_relu(double x, double alpha);
double elu(double x, double alpha);
double tanh_activation(double x);
double maxout(double x1, double x2);

void load_data(const char* filename, double** inputs, double outputs[], int* num_samples, int input_size);
void save_data(const char* filename, double** inputs, double outputs[], int num_samples);

void load_model(const char* filename, double** hidden_weights, double** output_weights, double hidden_bias[], double output_bias[], int hidden_size, int input_size, int output_size);
void save_model(const char* filename, double** hidden_weights, double** output_weights, double* hidden_bias, double* output_bias, int hidden_size, int input_size, int output_size);

void train(double** inputs, double outputs[], int num_samples, double** hidden_weights, double** output_weights, double hidden_bias[], double output_bias[], int input_size, int output_size, int hidden_size, int max_iterations);
void retrain(double** inputs, double outputs[], int num_samples, double** hidden_weights, double** output_weights, double hidden_bias[], double output_bias[], int input_size, int output_size, int hidden_size, int max_iterations);

double predict(double inputs[], double* hidden_weights[], double* output_weights[], double hidden_bias[], double output_bias[], int input_size, int hidden_size, int output_size);

