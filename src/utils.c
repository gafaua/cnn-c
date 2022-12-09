#include "utils.h"
#include "mnist.h"

void load_data() {
  load_mnist();
}

long long timeInMilliseconds(void) {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000)+(tv.tv_usec/1000);
}


void load_batch(Data2D* inputs, int* gt, int pos, int batch, float data[][784], int labels[], int* indices) {
    int start = pos*batch;

    #pragma omp parallel for 
    for (int i = 0; i < batch; i++) {
        int idx = indices[i+start];
        gt[i] = labels[idx];
        for (int j = 0; j < 28; j++)
            for (int k = 0; k < 28; k++)
                inputs->data[i][0].mat[j][k] = data[idx][k + j*28];
    }

}

void shuffle(int *arr, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        //srand(time(NULL));
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = arr[j];
          arr[j] = arr[i];
          arr[i] = t;
        }
    }
}

void train_epoch(Network* net, float lr, int* indices, int epoch, int batch_size) {
    int num_batch = NUM_TRAIN / batch_size;
    
    Data2D* inputs = CreateData2D(28, batch_size, 1);
    int* gt = (int*) malloc(sizeof(int) * batch_size);

    Data1D* outputs;
    LossResult loss;
    setbuf(stdout, NULL);

    float loss_sum = 0.0;
    float acc_sum = 0.0;
    float time_elapsed = 0;
    int cnt = 0;
    long long eta;
    int min, s;
    long long start = timeInMilliseconds();

    for (int i = 0; i < num_batch; i++) {
        printf("\r[%d] Train: Loading -> ", epoch);
        load_batch(inputs, gt, i, batch_size, train_image, train_label, indices);

        printf("Forward -> ");
        outputs = (Data1D*) network_forward(net, (DataType*) inputs);

        loss = CrossEntropy(net, outputs, gt);

        printf("Backward -> ");
        network_backward(net, (DataType*) loss.dL, lr);

        DestroyData1D(outputs);
        if (loss.value != INFINITY) {
            loss_sum += loss.value;
            acc_sum += loss.accuracy;
            cnt++;
        }

        time_elapsed = (timeInMilliseconds() - start) / 1000;
        eta = ((num_batch - i - 1) * (time_elapsed/(i+1)));
        printf(" [%d/%d] | [ETA %2d min %2d s > %2d min %2d s] | loss: %f (%f) | acc: %f (%f)", i, num_batch, (int)eta/60, (int)eta%60, (int)time_elapsed/60, (int)time_elapsed%60, loss.value, loss_sum / cnt, loss.accuracy, acc_sum / cnt);
    }
}


void test_epoch(Network* net, int epoch) {
    int batch = 100;
    int num_batch = NUM_TEST / batch;
    
    Data2D* inputs = CreateData2D(28, batch, 1);
    int* gt = (int*) malloc(sizeof(int) * batch);

    Data1D* outputs;
    LossResult loss;
    setbuf(stdout, NULL);

    float loss_sum = 0.0;
    float acc_sum = 0.0;
    float time_elapsed = 0;
    int cnt = 0;
    long long eta;
    int min, s;
    long long start = timeInMilliseconds();

    int* indices = (int*) malloc(sizeof(int)*NUM_TEST);
    for (int i = 0; i < NUM_TEST; i++) indices[i] = i;

    for (int i = 0; i < num_batch; i++) {
        printf("\r[%d] Test: Loading -> ", epoch);
        load_batch(inputs, gt, i, batch, test_image, test_label, indices);

        printf("Forward -> ");
        outputs = (Data1D*) network_forward(net, (DataType*) inputs);

        loss = CrossEntropy(net, outputs, gt);

        DestroyData1D(outputs);

        if (loss.value != INFINITY) {
            loss_sum += loss.value;
            acc_sum += loss.accuracy;
            cnt++;
        }

        time_elapsed = (timeInMilliseconds() - start) / 1000;
        eta = ((num_batch - i - 1) * (time_elapsed/(i+1)));
        printf(" [%d/%d] | [ETA %2d min %2d s > %2d min %2d s] | loss: %f (%f) | acc: %f (%f)", i, num_batch, (int)eta/60, (int)eta%60, (int)time_elapsed/60, (int)time_elapsed%60, loss.value, loss_sum / cnt, loss.accuracy, acc_sum / cnt);
    }
}

Data2D* LoadImage(char* name) {
    int length, width;
    float** mat = LoadImagePgm(name, &length, &width);
    assert(length == width && "This program only supports square images");

    // Norm
    float sum = 0.0;
    float std = 0.0;
    float mean = 0.0;
    float eps = 1e-7;

    for (int i=0; i<length; i++)
        for (int j=0; j<width; j++)
            sum += mat[i][j];

    mean = sum / (length*width);

    for (int i=0; i<length; i++)
        for (int j=0; j<width; j++) {
            std += powf(mat[i][j] - mean, 2);
            mat[i][j] -= mean;
        }

    std = sqrt(std+eps);

    for (int i=0; i<length; i++)
        for (int j=0; j<width; j++)
            mat[i][j] /= std;

    Data2D* d = (Data2D*) malloc(sizeof(Data2D));
    d->size = length;
    d->b = 1;
    d->c = 1;
    d->data = square_allocate_2d(1, 1);
    d->type = D2D;
    d->data[0][0].mat = mat;
    d->data[0][0].size = length;

    return d;
}

void SoftmaxTransform(Data1D* data) {
    float sum;
    float eps = 1e-7;
    for (int i = 0; i < data->b; i++) {
        sum = eps;
        for (int j = 0; j < data->n; j++) {
            data->mat[i][j] = exp(data->mat[i][j]);
            sum += data->mat[i][j];
        }

        for (int j = 0; j < data->n; j++)
            data->mat[i][j] /= sum;
    }
}


float** LoadImagePgm(char* name,int *length,int *width)
 {
  int i,j,k;
  unsigned char var;
  char buff[NBCHAR];
  float** mat;

  char stringTmp1[NBCHAR],stringTmp2[NBCHAR],stringTmp3[NBCHAR];
 
  int ta1,ta2,ta3;
  FILE *fic;

  /*-----nom du fichier pgm-----*/
  strcpy(buff,name);
  strcat(buff,".pgm");
  printf("---> Ouverture de %s",buff);

  /*----ouverture du fichier----*/
  fic=fopen(buff,"r");
  if (fic==NULL)
    { printf("\n- Grave erreur a l'ouverture de %s  -\n",buff);
      exit(-1); }

  /*--recuperation de l'entete--*/
  fgets(stringTmp1,100,fic);
  fgets(stringTmp2,100,fic);
  fscanf(fic,"%d %d",&ta1,&ta2);
  fscanf(fic,"%d\n",&ta3);

  /*--affichage de l'entete--*/
  printf("\n\n--Entete--");
  printf("\n----------");
  printf("\n%s%s%d %d \n%d\n",stringTmp1,stringTmp2,ta1,ta2,ta3);

  *length=ta1;
  *width=ta2;
  mat=fmatrix_allocate_2d(*length,*width);
   
  /*--chargement dans la matrice--*/
     for(i=0;i<*length;i++)
      for(j=0;j<*width;j++)  
        { fread(&var,1,1,fic);
          mat[i][j]=var; }

   /*---fermeture du fichier---*/
  fclose(fic);

  return(mat);
 }

/*----------------------------------------------------------*/
/* Sauvegarde de l'image de nom <name> au format pgm        */
/*----------------------------------------------------------*/
void SaveImagePgm(char* name,float** mat,int length,int width)
 {
  int i,j,k;
  char buff[NBCHAR];
  FILE* fic;
  time_t tm;

  /*--extension--*/
  strcpy(buff,name);
  strcat(buff,".pgm");

  /*--ouverture fichier--*/
  fic=fopen(buff,"w");
    if (fic==NULL) 
        { printf(" Probleme dans la sauvegarde de %s",buff); 
          exit(-1); }
  printf("\n Sauvegarde de %s au format pgm\n",name);

  /*--sauvegarde de l'entete--*/
  fprintf(fic,"P5");
  if (ctime(&tm)==NULL) fprintf(fic,"\n#\n");
  else fprintf(fic,"\n# IMG Module, %s",ctime(&tm));
  fprintf(fic,"%d %d",width,length);
  fprintf(fic,"\n255\n");

  /*--enregistrement--*/
     for(i=0;i<length;i++)
      for(j=0;j<width;j++) 
        fprintf(fic,"%c",(char)mat[i][j]);
   
  /*--fermeture fichier--*/
   fclose(fic); 
 } 
