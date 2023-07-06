/*
Author: Manuel Zarat
Date: 22.06.2023
License: MIT
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define LEARNING_RATE 0.01 // @todo make dynamic
#define MAX_ITERATIONS 10000  // @deprecated

/*
Sigmoid:
Die Sigmoid-Funktion, auch logistische Funktion genannt, ist eine Aktivierungsfunktion, die oft in neuronalen 
Netzwerken verwendet wird, insbesondere in früheren Modellen. Sie hat die Form f(x) = 1 / (1 + e^(-x)) und gibt 
Werte zwischen 0 und 1 aus. Sie wird häufig in binären Klassifikationsaufgaben verwendet, da sie eine 
Wahrscheinlichkeitsinterpretation ermöglicht. 
*/
__declspec (dllexport) double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/*
Rectified Linear Unit (ReLU): 
Die ReLU-Funktion ist eine weit verbreitete Aktivierungsfunktion, die für viele Anwendungen gute 
Ergebnisse liefert. Sie ist definiert als f(x) = max(0, x), d.h. sie gibt die Eingabe unverändert 
zurück, wenn sie positiv ist, andernfalls gibt sie 0 zurück. Die ReLU-Funktion ist einfach zu 
berechnen und kann zur Lösung des Vanishing-Gradient-Problems beitragen. 
*/
__declspec (dllexport) double relu(double x) {
    return (x > 0) ? x : 0;
}

/*
Leaky ReLU: 
Die Leaky ReLU-Funktion ist eine Variante der ReLU-Funktion, bei der eine kleine, konstante Steigung 
für negative Eingaben eingeführt wird, um das Problem der „tote Neuronen“ zu mildern. Es wird 
definiert als f(x) = max(ax, x), wobei a eine kleine positive Konstante ist. 
*/
__declspec (dllexport) double leaky_relu(double x, double alpha) {
    return (x > 0) ? x : alpha * x;
}

/*
Exponential Linear Unit (ELU): 
Die ELU-Funktion ist eine weitere Aktivierungsfunktion, die die Vorteile der ReLU-Funktion beibehält 
und auch für negative Eingaben einen sanften Übergang ermöglicht. Sie ist 
definiert als f(x) = x, wenn x > 0 und f(x) = a(e^x - 1), wenn x ? 0, wobei a eine positive Konstante ist. 
*/
__declspec (dllexport) double elu(double x, double alpha) {
    return (x > 0) ? x : alpha * (exp(x) - 1);
}

/*
Hyperbolic Tangent (Tanh): 
Die Tangens Hyperbolicus-Funktion ist eine S-förmige Aktivierungsfunktion, die Werte zwischen -1 und 1 ausgibt. 
Sie ist definiert als f(x) = (e^x - e^(-x)) / (e^x + e^(-x)). Die Tanh-Funktion ist symmetrisch um den Ursprung 
und erzeugt sowohl positive als auch negative Aktivierungen. 
*/ 
__declspec (dllexport) double tanh_activation(double x) {
    return tanh(x);
}

/*
MaxOut:
Die Maxout-Funktion verwendet einen Pool von Neuronen und gibt das Maximum der Aktivierungen in diesem Pool aus. 
Es findet eine „maximale Aktivierung“ statt, indem die Eingaben zu den Neuronen in mehrere Gruppen aufgeteilt werden 
und das Maximum der Aktivierungen ausgewählt wird. Mathematisch ausgedrückt kann die Maxout-Funktion 
als f(x) = max(w1x + b1, w2x + b2) dargestellt werden, wobei w1 und w2 die Gewichte, b1 und b2 die Bias-Termine sind.
*/
__declspec (dllexport) double maxout(double x1, double x2) {
    return fmax(x1, x2);
}

/*
Trainingsdaten aus Datei laden
*/
__declspec (dllexport) void load_data(const char* filename, double** inputs, double outputs[], int* num_samples, int input_size) {
    
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Fehler beim Öffnen der Datei '%s'.\n", filename);
        exit(1);
    }

    *num_samples = 0;
    double* input_row = malloc(sizeof(double) * input_size);

    while (fscanf(file, "%lf", &outputs[*num_samples]) == 1) {
        for (int i = 0; i < input_size; i++) {
            fscanf(file, "%lf", &input_row[i]);
        }

        inputs[*num_samples] = malloc(sizeof(double) * input_size);
        memcpy(inputs[*num_samples], input_row, sizeof(double) * input_size);

        (*num_samples)++;
    }

    fclose(file);
    free(input_row);
    
}

/*
Trainingsdaten in Datei speichern
@deprecated
*/
__declspec (dllexport) void save_data(const char* filename, double** inputs, double outputs[], int num_samples) {
    
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Fehler beim Öffnen der Datei '%s'.\n", filename);
        exit(1);
    }

    for (int i = 0; i < num_samples; i++) {
        fprintf(file, "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                inputs[i][0], inputs[i][1], inputs[i][2], inputs[i][3],
                inputs[i][4], inputs[i][5], inputs[i][6], inputs[i][7], 
                outputs[i]);
    }

    fclose(file);
    
}

/*
Modelldaten aus Datei laden.
*/
__declspec (dllexport) void load_model(const char* filename, double** hidden_weights, double** output_weights, double hidden_bias[], double output_bias[], int hidden_size, int input_size, int output_size) {
    
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Fehler beim Öffnen der Datei '%s'.\n", filename);
        exit(1);
    }

    for (int i = 0; i < hidden_size; i++) {
        
        hidden_weights[i] = malloc(sizeof(double) * input_size);
        
        for (int j = 0; j < input_size; j++) 
            fscanf(file, "%lf", &hidden_weights[i][j]);
        
    }

    for (int i = 0; i < output_size; i++) {
        
        output_weights[i] = malloc(sizeof(double) * hidden_size);
        
        for (int j = 0; j < hidden_size; j++) 
            fscanf(file, "%lf", &output_weights[i][j]);
        
    }

    for (int i = 0; i < hidden_size; i++) 
        fscanf(file, "%lf", &hidden_bias[i]);

    for (int i = 0; i < output_size; i++) 
        fscanf(file, "%lf", &output_bias[i]);

    fclose(file);
    
}

/*
Modelldaten in Datei speichern
*/
__declspec (dllexport) void save_model(const char* filename, double** hidden_weights, double** output_weights, double* hidden_bias, double* output_bias, int hidden_size, int input_size, int output_size) {
    
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Fehler beim Öffnen der Datei '%s'.\n", filename);
        exit(1);
    }

    for (int i = 0; i < hidden_size; i++) {
    
        for (int j = 0; j < input_size; j++) 
            fprintf(file, "%lf ", hidden_weights[i][j]);
        
        fprintf(file, "\n");
        
    }

    for (int i = 0; i < output_size; i++) {
        
        for (int j = 0; j < hidden_size; j++) 
            fprintf(file, "%lf ", output_weights[i][j]);
        
        fprintf(file, "\n");
        
    }

    for (int i = 0; i < hidden_size; i++) 
        fprintf(file, "%lf ", hidden_bias[i]);
    
    fprintf(file, "\n");

    for (int i = 0; i < output_size; i++) 
        fprintf(file, "%lf ", output_bias[i]);
    
    fprintf(file, "\n");

    fclose(file);
    
}

/* 
Ein neues Modell trainieren
*/
__declspec (dllexport) void train(double** inputs, double outputs[], int num_samples, double** hidden_weights, double** output_weights, double hidden_bias[], double output_bias[], int input_size, int output_size, int hidden_size, int max_iterations) {

    // Zufällige Initialisierung der Gewichte und Biases
    for (int i = 0; i < hidden_size; i++) {
        
        hidden_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        
        for (int j = 0; j < input_size; j++) 
            hidden_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        
    }

    for (int i = 0; i < output_size; i++) {
        
        output_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        
        for (int j = 0; j < hidden_size; j++) 
            output_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        
    }

    // Trainingsschleife
    for (int iteration = 0; iteration < max_iterations; iteration++) {

        double total_error = 0.0;

        // Durchlaufen der Trainingsdaten
        for (int sample = 0; sample < num_samples; sample++) {
            
            // Vorwärtsdurchgang
            double* hidden_layer = malloc(sizeof(double) * hidden_size);
            double* output_layer = malloc(sizeof(double) * output_size);

            for (int i = 0; i < hidden_size; i++) {
                
                double sum = hidden_bias[i];
                
                for (int j = 0; j < input_size; j++) {
                
                    sum += hidden_weights[i][j] * inputs[sample][j];
                    
                }
                
                hidden_layer[i] = sigmoid(sum);
                
            }

            for (int i = 0; i < output_size; i++) {
                
                double sum = output_bias[i];
                
                for (int j = 0; j < hidden_size; j++) {
                
                    sum += output_weights[i][j] * hidden_layer[j];
                    
                }
                
                output_layer[i] = sigmoid(sum);
                
            }

            // Berechnung des Fehlers
            double error = outputs[sample] - output_layer[0];
            total_error += fabs(error);

            // Rückwärtsdurchgang
            double output_delta = error * output_layer[0] * (1.0 - output_layer[0]);

            for (int i = 0; i < hidden_size; i++) {
                
                double hidden_delta = hidden_layer[i] * (1.0 - hidden_layer[i]) * output_delta * output_weights[0][i];

                for (int j = 0; j < input_size; j++) {
                
                    hidden_weights[i][j] += LEARNING_RATE * hidden_delta * inputs[sample][j];
                    
                }
                
                hidden_bias[i] += LEARNING_RATE * hidden_delta;
                
            }

            for (int i = 0; i < output_size; i++) {
            
                for (int j = 0; j < hidden_size; j++) {
                
                    output_weights[i][j] += LEARNING_RATE * output_delta * hidden_layer[j];
                    
                }
                
                output_bias[i] += LEARNING_RATE * output_delta;
                
            }

            free(hidden_layer);
            free(output_layer);
            
        }

        double _x = (double)iteration / (double)max_iterations;
        double percentage = _x * 100.0;
        
        // Ausgabe des aktuellen Fehlers
        printf("Iteration: %d (%.2f% %), Error: %.16f\n", iteration, percentage, total_error);
        
        // Abbruch, wenn der Fehler klein genug ist
        if (total_error < 0.01) 
            break;       
        
    }
    
}

/*
Mit vorhandenen Modelldaten weiterlernen.
*/
__declspec (dllexport) void retrain(double** inputs, double outputs[], int num_samples, double** hidden_weights, double** output_weights, double hidden_bias[], double output_bias[], int input_size, int output_size, int hidden_size, int max_iterations) {

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        
        //clock_t start_time, end_time;
        //double cpu_time_used;
        //start_time = clock();
        
        double total_error = 0.0;

        // Durchlaufen der Trainingsdaten
        for (int sample = 0; sample < num_samples; sample++) {
            
            // Vorwärtsdurchgang
            double* hidden_layer = malloc(sizeof(double) * hidden_size);
            double* output_layer = malloc(sizeof(double) * output_size);

            for (int i = 0; i < hidden_size; i++) {
            
                double sum = hidden_bias[i];
                
                for (int j = 0; j < input_size; j++) 
                    sum += hidden_weights[i][j] * inputs[sample][j];
                
                hidden_layer[i] = sigmoid(sum);
                
            }

            for (int i = 0; i < output_size; i++) {
            
                double sum = output_bias[i];
                
                for (int j = 0; j < hidden_size; j++) 
                    sum += output_weights[i][j] * hidden_layer[j];
                
                output_layer[i] = sigmoid(sum);
                
            }

            // Berechnung des Fehlers
            double error = outputs[sample] - output_layer[0];
            total_error += fabs(error);

            // Rückwärtsdurchgang
            double output_delta = error * output_layer[0] * (1.0 - output_layer[0]);

            for (int i = 0; i < hidden_size; i++) {
                
                double hidden_delta = hidden_layer[i] * (1.0 - hidden_layer[i]) * output_delta * output_weights[0][i];

                for (int j = 0; j < input_size; j++) 
                    hidden_weights[i][j] += LEARNING_RATE * hidden_delta * inputs[sample][j];
                
                hidden_bias[i] += LEARNING_RATE * hidden_delta;
                
            }

            for (int i = 0; i < output_size; i++) {
            
                for (int j = 0; j < hidden_size; j++) 
                    output_weights[i][j] += LEARNING_RATE * output_delta * hidden_layer[j];
                
                output_bias[i] += LEARNING_RATE * output_delta;
                
            }

            free(hidden_layer);
            free(output_layer);
            
        }
        
        //end_time = clock();
        //cpu_time_used = ((double)(end_time - start_time))/CLOCKS_PER_SEC;
        
        double _x = (double)iteration / (double)max_iterations;
        double percentage = _x * 100.0;
        
        // Ausgabe des aktuellen Fehlers
        printf("Iteration: %d (%.2f% %), Error: %.16f\n", iteration, percentage, total_error);
        
        // Abbruch, wenn der Fehler klein genug ist
        if (total_error < 0.01) 
            break;
        
    }
    
}

/*
Einem Modell eine Frage stellen.
*/
__declspec (dllexport) double predict(double inputs[], double* hidden_weights[], double* output_weights[], double hidden_bias[], double output_bias[], int input_size, int hidden_size, int output_size) {

    double* hidden_layer = malloc(sizeof(double) * hidden_size);
    double* output_layer = malloc(sizeof(double) * output_size);

    // Vorwärtsdurchgang
    for (int i = 0; i < hidden_size; i++) {
        double sum = hidden_bias[i];
        for (int j = 0; j < input_size; j++) {
            sum += hidden_weights[i][j] * inputs[j];
        }
        hidden_layer[i] = sigmoid(sum);
    }

    for (int i = 0; i < output_size; i++) {
        double sum = output_bias[i];
        for (int j = 0; j < hidden_size; j++) {
            sum += output_weights[i][j] * hidden_layer[j];
        }
        output_layer[i] = sigmoid(sum);
    }

    double prediction = output_layer[0];

    free(hidden_layer);
    free(output_layer);

    return prediction;
    
}

