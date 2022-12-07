#include "tests.h"
#include <float.h>

#define TOLERANCE 1e-5

void test_all() {
    //test_functions_memory();
    test_network();
    test_mnist_network();
    test_conv2d_forward_backward();
    test_linear_forward_backward();
}


void test_functions_memory() {
    int n = 10;
    int b = 2;

    int size2d = 5;
    int c = 2;

    Data1D* d1d = CreateData1D(n, b);
    random_init_matrix(d1d->mat, b, n);
    Data2D* d2d = CreateData2D(size2d, b, c);
    RandomInitData2D(d2d);
    printf("Creating Linear...");
    LinearLayer* ll = CreateLinearLayer(n, 5, TRUE, TRUE);
    printf("Ok\nRandom Linear Init...");
    printf("Ok\nCreating Conv...");
    ConvLayer* cl = CreateConvLayer(c, 1, 3, TRUE, TRUE);
    printf("Ok\nTesting LinearLayer forward...");
    Data1D* d1d_y = linear_forward(ll, d1d);
    printf("Ok\nTesting Linear Backward...");
    d1d_y->mat[0][0] = -1;
    Data1D* d1d_y_ = linear_backward(ll, d1d_y, 0.000001);
    printf("Ok\nTesting print Data1d...");
    print_data1d(d1d_y_);
    printf("Ok\nTesting Conv Forward...");
    Data2D* d2d_y = conv_forward(cl, d2d);
    printf("Ok\nTesting Conv Backward...");
    Data2D* d2d_y_ = conv_backward(cl, d2d_y, 0.000001);
    print_data2d(d2d_y_);
    printf("Ok\nTesting Flatten...");
    Data1D* d2d_flat = flatten(d2d_y_);
    printf("Ok\nTesting Unflatten...");
    Data2D* d1d_unflat = unflatten(d2d_flat, c);

    printf("Ok\nDestroying Data1D...");
    DestroyData1D(d1d_y);
    DestroyData1D(d1d_y_);
    DestroyData1D(d2d_flat);
    printf("Ok\nDestroying Data2D...");
    DestroyData2D(d2d_y);
    DestroyData2D(d2d_y_);
    DestroyData2D(d1d_unflat);
    printf("Ok\nDestroying Linear Layer...");
    DestroyLinearLayer(ll);
    printf("Ok\nDestroying Conv Layer...");
    DestroyConvLayer(cl);
    printf("\nOk\n");

    printf("Individual functions are memory safe\n");
}

void test_network() {
    int in = 10;
    int b = 2;

    printf("Testing Network Creation...");
    Network* net = CreateNetwork();
    printf("Ok\nTesting Add to network...");
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(in, 100, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(100, 500, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(500, 100, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(100, 32, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(32, 200, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateUnflattenLayer(5));
    AddToNetwork(net, (LayerNode*) CreateConvLayer(2, 5, 3, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateConvLayer(5, 2, 3, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateConvLayer(2, 5, 3, TRUE, TRUE));
    AddToNetwork(net, (LayerNode*) CreateFlattenLayer(5));
    AddToNetwork(net, (LayerNode*) CreateLinearLayer(80, 10, TRUE, TRUE));
    printf("Ok\nTesting Forward pass...");

    Data1D* inputs = CreateData1D(in, b);
    random_init_matrix(inputs->mat, b, in);
    Data1D* outputs;
    Data1D* dY;
    int epochs = 2;
    for (int i = 1; i <=epochs; i++) {
        printf("Ok\nTesting Forward pass %d/%d...", i, epochs);
        outputs = (Data1D*) network_forward(net, (DataType*) inputs);
        dY = CreateData1D(outputs->n, outputs->b);
        init_matrix(dY->mat, 1.0, dY->b ,dY->n);
        printf("Ok\nTesting Backward pass %d/%d...", i, epochs);

        network_backward(net, (DataType*) dY, 0.00001);
        DestroyData1D(outputs);
    }

    printf("Ok\nTesting Destroy Network...");

    DestroyData1D(inputs);
    DestroyNetwork(net);
    printf("Ok\nNetwork passed all tests.\n");
}

void test_mnist_network() {
    printf("Testing MNIST network Creation...\n");
    Network* net = CreateNetworkMNIST(TRUE);
    int batch = 16;
    printf("Gt: ");
    int gt[batch];
    for (int i = 0; i < batch; i++) {
        gt[i] = rand() % 10;
        printf("%d ", gt[i]);
    }

    Data2D* inputs = CreateData2D(28, batch, 1);
    RandomInitData2D(inputs);

    int num_batch = 300;
    Data1D* outputs;
    setbuf(stdout, NULL);

    for (int i = 1; i <=num_batch; i++) {
        printf("\nTesting Forward pass %d/%d...", i, num_batch);
        outputs = (Data1D*) network_forward(net, (DataType*) inputs);
        //print_data1d(outputs);
        LossResult loss = CrossEntropy(net, outputs, gt);
        printf(" Loss: %f", loss.value);
        printf("\nTesting Backward pass %d/%d...", i, num_batch);
        network_backward(net, (DataType*) loss.dL, 0.000001);
        print_data1d(outputs);
        DestroyData1D(outputs);
    }

    for (int i = 0; i < batch; i++) {
        printf("%d ", gt[i]);
    }

    DestroyData2D(inputs);
    DestroyNetwork(net);
    printf("\nTesting MNIST network OK...\n");
}

int equals_with_tolerance(float expected, float value) {
    float diff = fabsf(expected - value);
    float deviation = diff / expected;
    if (deviation > TOLERANCE) {
        printf("Equals check failed, difference: %.*e, deviation: %.*e%%  \n", DECIMAL_DIG, diff, DECIMAL_DIG, deviation);
        return FALSE;
    }

    return TRUE;
}

void test_conv2d_forward_backward() {
    // Test data
    int ins = 2;
    int outs = 3;
    int batch = 2;
    int i_size = 5;
    int o_size = 3;
    int k_size = 3;
    // Data calculated using torch
    float inputs[2][2][5][5] = {{{{0.383707761764526, 0.487678289413452, 0.424733161926270,0.158605158329010, 0.285557270050049},{0.290969312191010, 0.185832381248474, 0.406805574893951,0.491074562072754, 0.354025661945343},{0.677513122558594, 0.466112792491913, 0.398037493228912,0.081802606582642, 0.630491256713867},{0.657838642597198, 0.565780282020569, 0.957729518413544,0.306446671485901, 0.915569424629211},{0.363453388214111, 0.972853481769562, 0.749442100524902,0.659345805644989, 0.312170505523682}},{{0.467018425464630, 0.390613615512848, 0.609391748905182,0.402547955513000, 0.912592053413391},{0.262436509132385, 0.808416187763214, 0.074977576732635,0.939343035221100, 0.864800274372101},{0.252142965793610, 0.585045874118805, 0.896523833274841,0.796870648860931, 0.477829337120056},{0.522816121578217, 0.389267504215240, 0.300254464149475,0.231149315834045, 0.981493473052979},{0.348074793815613, 0.927018642425537, 0.113915979862213,0.927521228790283, 0.211132109165192}}},{{{0.914300560951233, 0.945284307003021, 0.230241715908051,0.799951851367950, 0.717741370201111},{0.873918116092682, 0.270160317420959, 0.792392671108246,0.249032139778137, 0.146237313747406},{0.424271106719971, 0.263534903526306, 0.833154976367950,0.906268656253815, 0.154842972755432},{0.066637575626373, 0.314704775810242, 0.123976945877075,0.412196636199951, 0.597157716751099},{0.175544261932373, 0.912800669670105, 0.592240929603577,0.354016840457916, 0.481846272945404}},
{{0.800443410873413, 0.561971664428711, 0.951985955238342,0.832737445831299, 0.217283606529236},{0.842395544052124, 0.287329554557800, 0.929738759994507,0.231413483619690, 0.645466983318329},{0.913046002388000, 0.630163788795471, 0.333611965179443,0.630807816982269, 0.520674049854279},{0.391171813011169, 0.307950556278229, 0.761068880558014,0.122177362442017, 0.938039302825928},{0.359409034252167, 0.676563262939453, 0.085195600986481,0.139181792736053, 0.294080376625061}}}};

    float weights[3][2][3][3] = {{{{-0.027261929586530,  0.179684042930603,  0.235266074538231},{-0.084773443639278, -0.043336682021618,  0.081139147281647},{-0.166239932179451, -0.093605987727642, -0.166601777076721}},
{{ 0.102728247642517,  0.006250546313822,  0.142596691846848},{ 0.113512031733990,  0.091408535838127, -0.070863015949726},{-0.206827342510223, -0.230891764163971,  0.052758384495974}}},
{{{-0.055539432913065,  0.001749599934556, -0.076913028955460},{-0.168513968586922,  0.051149077713490, -0.037276949733496},{ 0.027956735342741,  0.154750838875771,  0.122027009725571}},
{{-0.130070626735687,  0.124964453279972, -0.144426718354225},{-0.051762007176876, -0.019880611449480,  0.112848296761513},{-0.059161785990000,  0.160354092717171, -0.048149403184652}}},
{{{-0.192984327673912,  0.018565518781543, -0.157088607549667},{-0.104908816516399, -0.014636420644820,  0.232882663607597},{-0.039620228111744, -0.032021630555391,  0.085709609091282}},
{{-0.012061276473105, -0.085196621716022,  0.161492556333542},{-0.005121150985360,  0.203169733285904,  0.004652366042137},{-0.001592954155058,  0.107297644019127, -0.184248089790344}}}};

    float expected_outputs[2][3][3][3] = {{{{ 0.050560645759106, -0.174144327640533, -0.237672761082649},{-0.366948187351227,  0.070389963686466,  0.047822717577219},{-0.168963819742203, -0.262783974409103, -0.182018414139748}},
{{-0.053582940250635,  0.117909051477909, -0.035175178200006},{ 0.220105022192001, -0.082835771143436, -0.002985484898090},{ 0.099605888128281, -0.026052167639136,  0.047183848917484}},
{{ 0.042063768953085, -0.074149414896965,  0.240602269768715},{-0.016479207202792,  0.139477834105492,  0.071329407393932},{ 0.227430835366249, -0.158262476325035,  0.018762867897749}}},
{{{-0.095656722784042,  0.003528096480295,  0.018844814971089},{ 0.416824162006378, -0.031222186982632, -0.086970299482346},{-0.052849624305964,  0.191346004605293,  0.094074338674545}},
{{-0.160318627953529,  0.023183407261968,  0.014913861639798},{-0.372882723808289,  0.185195982456207, -0.241401880979538},{ 0.185941219329834, -0.166302263736725,  0.078861579298973}},
{{ 0.099282711744308, -0.091539293527603, -0.274668276309967},{-0.011937650851905,  0.194399908185005, -0.161558181047440},{-0.066409572958946,  0.060506355017424, -0.029536092653871}}}};

    float expected_dW[3][2][3][3] = {{{{ 9.268649101257324,  8.390704154968262,  8.060996055603027},{ 8.569370269775391,  8.025043487548828,  8.757242202758789},{ 9.515626907348633,  9.870445251464844,  9.466736793518066}},
{{10.597253799438477, 10.893490791320801, 11.268596649169922},{ 9.488357543945312,  9.256111145019531, 10.676240921020508},{ 8.793240547180176,  8.854288101196289,  8.761527061462402}}},
{{{ 9.268649101257324,  8.390704154968262,  8.060996055603027},{ 8.569370269775391,  8.025043487548828,  8.757242202758789},{ 9.515626907348633,  9.870445251464844,  9.466736793518066}},
{{10.597253799438477, 10.893490791320801, 11.268596649169922},{ 9.488357543945312,  9.256111145019531, 10.676240921020508},{ 8.793240547180176,  8.854288101196289,  8.761527061462402}}},
{{{ 9.268649101257324,  8.390704154968262,  8.060996055603027},{ 8.569370269775391,  8.025043487548828,  8.757242202758789},{ 9.515626907348633,  9.870445251464844,  9.466736793518066}},
{{10.597253799438477, 10.893490791320801, 11.268596649169922},{ 9.488357543945312,  9.256111145019531, 10.676240921020508},{ 8.793240547180176,  8.854288101196289,  8.761527061462402}}}};

    float expected_dX[2][2][5][5] = {{{{-0.275785684585571, -0.075786523520947, -0.074522078037262,  0.201263606548309,  0.001264438033104},{-0.633981943130493, -0.440806806087494, -0.162797525525093,  0.471184432506561,  0.278009295463562},{-0.811885356903076, -0.589587032794952, -0.270442903041840,  0.541442453861237,  0.319144129753113},{-0.536099672317505, -0.513800501823425, -0.195920795202255,  0.340178847312927,  0.317879706621170},{-0.177903413772583, -0.148780196905136, -0.107645355165005,  0.070258058607578,  0.041134841740131}},
{{-0.039403654634953,  0.006614722311497,  0.166277244687080,  0.205680921673775,  0.159662529826164},{ 0.017225218936801,  0.337941259145737,  0.544241428375244,  0.527016222476959,  0.206300184130669},{-0.250356853008270,  0.107119143009186,  0.133780211210251,  0.384137094020844,  0.026661068201065},{-0.210953190922737,  0.100504428148270, -0.032497033476830,  0.178456187248230, -0.133001461625099},{-0.267582088708878, -0.230822101235390, -0.410461217164993, -0.142879143357277, -0.179639101028442}}},
{{{-0.275785684585571, -0.075786523520947, -0.074522078037262,  0.201263606548309,  0.001264438033104},{-0.633981943130493, -0.440806806087494, -0.162797525525093,  0.471184432506561,  0.278009295463562},{-0.811885356903076, -0.589587032794952, -0.270442903041840,  0.541442453861237,  0.319144129753113},{-0.536099672317505, -0.513800501823425, -0.195920795202255,  0.340178847312927,  0.317879706621170},{-0.177903413772583, -0.148780196905136, -0.107645355165005,  0.070258058607578,  0.041134841740131}},
{{-0.039403654634953,  0.006614722311497,  0.166277244687080,  0.205680921673775,  0.159662529826164},{ 0.017225218936801,  0.337941259145737,  0.544241428375244,  0.527016222476959,  0.206300184130669},{-0.250356853008270,  0.107119143009186,  0.133780211210251,  0.384137094020844,  0.026661068201065},{-0.210953190922737,  0.100504428148270, -0.032497033476830,  0.178456187248230, -0.133001461625099},{-0.267582088708878, -0.230822101235390, -0.410461217164993, -0.142879143357277, -0.179639101028442}}}};

    printf("Testing loading test weights and test data...");
    ConvLayer* layer = CreateConvLayer(ins, outs, k_size, TRUE, FALSE);
    Data2D* data_input = CreateData2D(i_size, batch, ins);

    int i, j, k, l;
    for (i=0; i<layer->out; i++)
        for (j=0; j<layer->in; j++)
            for(k=0; k<layer->size; k++)
                for(l=0; l<layer->size; l++)
                    layer->w[i][j].mat[k][l] = weights[i][j][k][l];
    
    for(i=0; i<batch; i++)
        for(j=0; j<ins; j++)
            for(k=0; k<i_size; k++)
                for(l=0; l<i_size; l++)
                    data_input->data[i][j].mat[k][l] = inputs[i][j][k][l];
    
    printf("OK\nTesting forward pass...\n");

    Data2D* data_output = conv_forward(layer, data_input);

    for(i=0; i<batch; i++)
        for(j=0; j<outs; j++)
            for(k=0; k<o_size; k++)
                for(l=0; l<o_size; l++)
                    assert(equals_with_tolerance(expected_outputs[i][j][k][l], data_output->data[i][j].mat[k][l]) && "Output not equals to expected output after forward pass");

    printf("Ok\nTesting backward pass...\n");

    for(i=0; i<batch; i++)
        for(j=0; j<outs; j++)
            for(k=0; k<o_size; k++)
                for(l=0; l<o_size; l++)
                    data_output->data[i][j].mat[l][k] = 1.0;

    Data2D* dX = conv_backward(layer, data_output, 0.0);

    for (i=0; i<outs; i++) 
        for (j=0; j<ins; j++)
            for(k=0; k<k_size; k++)
                for(l=0; l<k_size; l++)
                    assert(equals_with_tolerance(expected_dW[i][j][k][l], layer->dW[i][j].mat[k][l]) && "Gradient w.r.t the weights not equals to expected values after backwards pass");

    for(i=0; i<batch; i++)
        for(j=0; j<ins; j++)
            for(k=0; k<i_size; k++)
                for(l=0; l<i_size; l++)
                    assert(equals_with_tolerance(expected_dX[i][j][k][l], dX->data[i][j].mat[k][l])  && "Gradient w.r.t the input not equals to expected values after backwards pass");

    printf("Ok\n");

    DestroyConvLayer(layer);
    DestroyData2D(data_output);
    DestroyData2D(dX);
}

void test_linear_forward_backward() {
    int i, j;
    int ins = 10;
    int batch = 2;
    int outs = 20;
    float inputs[2][10] = {{0.8472404480, 0.0313712955, 0.9778280854, 0.8820483685, 0.0894103050,0.4225116968, 0.2173921466, 0.7278696895, 0.8558441401, 0.0733106732},{0.9509444237, 0.0117841363, 0.7386362553, 0.4075754285, 0.7741430998,0.3023173809, 0.0386329889, 0.1873695850, 0.6026180387, 0.8633670807}};
    float weights[20][10] = {{ 1.5697236359e-01, -7.2457842529e-02,  2.2459255159e-01,2.9851418734e-01,  3.1940422952e-02, -2.7329188585e-01,-1.7873086035e-01, -2.6484229602e-03, -9.1424100101e-02,2.0851798356e-01},{-1.4449921250e-01,  5.7005308568e-02,  2.4027018249e-01,-2.4247346818e-01,  1.0218931735e-01, -1.0020466894e-01,-1.9591362774e-01,  5.4221551865e-02,  6.5842078766e-04,2.6106154919e-01},{ 2.3500429094e-01,  1.4295135438e-01, -1.1113405228e-01,-1.2802910805e-01, -5.0877541304e-02,  1.6169184446e-01,-2.4611514807e-01,  1.5198287368e-01,  2.4901048839e-01,2.7862995863e-01},{ 3.0808967352e-01, -6.7167326808e-02, -2.6129624248e-01,1.3847167790e-01, -1.0185302049e-01, -4.6546724625e-03,2.3940895498e-01,  3.1603896618e-01,  1.2902906537e-01,-4.3911382556e-02},{-1.9193384796e-02, -2.7720248699e-01,  1.1818137020e-02,-1.2907716632e-01,  1.6501863301e-01,  3.6616805941e-02,3.0783832073e-01, -2.1683573723e-01,  1.1643361300e-01,1.5463050455e-02},{ 2.8085711598e-01,  2.2842672188e-03, -3.1143119931e-01,1.7479975522e-01,  5.0510331988e-02, -1.0974343866e-01,4.4074613601e-02,  6.2176775187e-02,  1.9291220605e-01,8.4057860076e-02},{ 1.4805258811e-01,  3.3325117081e-02,  7.5097441673e-02,1.9250167906e-01,  3.0357399583e-01,  1.1494094878e-01,2.9333594441e-01,  2.0982544124e-01, -1.0614025593e-01,1.1430345476e-02},{ 2.2965945303e-01, -1.8330766261e-01, -4.2733643204e-02,1.1314225942e-01,  2.3297654092e-01, -5.9560243040e-02,-1.2225136161e-01,  2.3409423232e-01,  2.9432533775e-03,-5.5033665150e-02},{ 2.9768453911e-02, -2.1333388984e-01,  1.4727199450e-03,-2.9112827778e-01,  5.3276181221e-02,  1.9339126348e-01,-1.5669789910e-01, -3.0090340972e-01,  7.4492588639e-02,-2.4093873799e-01},{-1.9548796117e-02,  2.9251191020e-01, -1.7814297974e-01,-1.4537517726e-01, -2.4041132629e-01, -2.9897508025e-01,2.2483713925e-01, -2.3732206225e-01,  2.9479211569e-01,5.8669492602e-02},{ 1.9001106918e-01, -4.6340294182e-02, -2.6466986537e-01,2.6754176617e-01,  1.8821614981e-01, -1.2668742239e-01,-3.1511330605e-01,  1.4950382710e-01,  1.0490834713e-01,2.0827785134e-01},{ 8.7836377323e-02, -9.2372678220e-02, -3.0809653923e-02,3.1483781338e-01,  1.4873155951e-01,  1.9050090015e-01,2.1996955574e-01,  2.2465348244e-01,  1.2999092042e-01,5.9972539544e-02},{-6.8330630660e-02,  9.7022339702e-02,  1.5702164173e-01,-2.6022523642e-01,  9.6644312143e-02,  2.2686491907e-01,-5.0990633667e-02,  2.9124361277e-01, -8.3714932203e-02,-2.8291025758e-01},{-2.8896787763e-01, -2.9400277138e-01, -2.3018853366e-01,-2.8637972474e-01, -1.9278818369e-01,  3.0530351400e-01,-2.2984628379e-01, -9.6407383680e-02,  1.1346551776e-01,1.2900799513e-01},{-2.1496237814e-01, -1.2668670714e-01,  1.2113231421e-01,1.7389686406e-01,  2.3215425014e-01,  2.9568418860e-01,5.0345029682e-02,  2.4707408249e-01, -1.8953660131e-01,7.9587303102e-02},{-2.5804162025e-01,  2.6289032772e-02,  6.2057763338e-02,-3.1597865745e-04, -6.2279123813e-02,  6.4768046141e-02,-2.0068436861e-01, -1.2230673432e-01,  4.0445834398e-02,9.7982783336e-04},{-3.1842144672e-03,  1.2494222820e-01,  1.3433159888e-01,2.2012726963e-01, -4.4729564339e-02,  2.7525329590e-01,2.5635603070e-01, -2.1036714315e-01,  1.6333955526e-01,2.6174247265e-01},{-1.9386745989e-01,  2.6114752889e-01,  4.7447313555e-03,1.0247091204e-01, -3.8701165468e-03, -2.0387732983e-01,2.4413761497e-01,  1.0210261494e-01, -2.9629695415e-01,2.9757064581e-01},{ 3.0571129918e-01, -1.5882225335e-01,  1.3225878775e-01,3.0681687593e-01,  7.2452791035e-02, -8.1074990332e-02,4.2264539748e-02,  1.4834944904e-01, -1.2813065946e-01,-9.8745403811e-03},{-2.8613480926e-01, -2.9687270522e-01,  5.6867789477e-02,-6.4651034772e-02, -2.2671039402e-01, -1.0997124016e-01,2.7152913809e-01, -1.7416666448e-01, -2.7852895856e-01,1.1348296702e-01}};
    float expected_outputs[2][20] = {{ 0.3972833455, -0.1161902547,  0.3364205062,  0.5037719011,-0.0871522874,  0.2719322741,  0.5719093084,  0.3848006129,-0.3573399186, -0.3248874247,  0.2453863323,  0.7398700118,0.0821693838, -0.6335493326,  0.2658230364, -0.2335620075,0.5006279945, -0.2519004643,  0.6329883337, -0.6176912785},{ 0.4956155717,  0.2191020846,  0.5099741220,  0.1836722046,0.1281996965,  0.3164189458,  0.5414233804,  0.3865323663,-0.2176276445, -0.2899556756,  0.4599775970,  0.5414378047,-0.1526163816, -0.4691368341,  0.2262837440, -0.2334468216,0.5308703780, -0.0937207341,  0.4868532419, -0.5607074499}};
    float expected_dW[20][10] = {{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539},{1.7981848717, 0.0431554317, 1.7164642811, 1.2896237373, 0.8635534048,0.7248290777, 0.2560251355, 0.9152392745, 1.4584622383, 0.9366777539}};
    float expected_dX[2][10] = {{ 0.4752323627, -0.7910874486, -0.2087405324,  0.7554659247,0.7541652322,  0.4969747066,  0.4977534115,  0.8303093314,0.4386495352,  1.4357831478},{ 0.4752323627, -0.7910874486, -0.2087405324,  0.7554659247,0.7541652322,  0.4969747066,  0.4977534115,  0.8303093314,0.4386495352,  1.4357831478}};

    printf("Testing Linear Forward Backward...\n");
    printf("Loading weights and inputs...\n");

    LinearLayer* layer = CreateLinearLayer(ins, outs, TRUE, FALSE);
    Data1D* input = CreateData1D(ins, batch);

    for (i = 0; i < outs; i++)
        for (j = 0; j < ins; j++)
            layer->w[i][j] = weights[i][j];
    
    for (i = 0; i < batch; i++)
        for (j = 0; j < ins; j++)
            input->mat[i][j] = inputs[i][j];
    
    printf("Ok\nTesting Forward pass...");
    Data1D* output = linear_forward(layer, input);

    for(i=0; i<batch; i++)
        for(j=0; j<outs; j++)
            assert(equals_with_tolerance(expected_outputs[i][j], output->mat[i][j]) && "Output not equals to expected output after forward pass");

    printf("Ok\nTesting Backward pass...");

    Data1D* dY = CreateData1D(outs, batch);
    init_matrix(dY->mat, 1.0, batch, outs);

    Data1D* dX = linear_backward(layer, dY, 0.0);

    for(i=0; i<outs; i++)
        for(j=0; j<ins; j++)
            assert(equals_with_tolerance(expected_dW[i][j], layer->dW[i][j]) && "Gradient w.r.t the weights not equals to expected values after backward pass");

    for (i = 0; i < batch; i++)
        for (j = 0; j < ins; j++)
            assert(equals_with_tolerance(expected_dX[i][j], dX->mat[i][j]) && "Gradient w.r.t the input not equals to expected values after backward pass");

    printf("Ok\nLinear Layer forward and backward passes are accurate");
    
    DestroyData1D(dX);
    DestroyData1D(dY);
    DestroyData1D(output);
    DestroyLinearLayer(layer);
}
