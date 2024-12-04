!!!!test_begin!!!!

#include <iostream>
#include <cassert>

// Test case for SSMScanParamsBase
void test_SSMScanParamsBase() {
    // Happy path
    SSMScanParamsBase params1;
    params1.batch = 2;
    params1.seqlen = 3;
    params1.n_chunks = 4;
    params1.a_batch_stride = 5;
    params1.b_batch_stride = 6;
    params1.out_batch_stride = 7;
    params1.a_ptr = nullptr;
    params1.b_ptr = nullptr;
    params1.out_ptr = nullptr;
    params1.x_ptr = nullptr;

    assert(params1.batch == 2);
    assert(params1.seqlen == 3);
    assert(params1.n_chunks == 4);
    assert(params1.a_batch_stride == 5);
    assert(params1.b_batch_stride == 6);
    assert(params1.out_batch_stride == 7);

    // Negative case
    SSMScanParamsBase params2;
    params2.batch = -1;

    try {
        assert(false); // Should not reach here
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
}

// Test case for SSMParamsBase
void test_SSMParamsBase() {
    // Happy path
    SSMParamsBase params1;
    params1.batch = 2;
    params1.dim = 3;
    params1.seqlen = 4;
    params1.dstate = 5;
    params1.n_groups = 6;
    params1.n_chunks = 7;
    params1.dim_ngroups_ratio = 8;
    params1.delta_softplus = true;
    params1.A_d_stride = 9;
    params1.A_dstate_stride = 10;
    params1.B_batch_stride = 11;
    params1.B_d_stride = 12;
    params1.B_dstate_stride = 13;
    params1.B_group_stride = 14;
    params1.C_batch_stride = 15;
    params1.C_d_stride = 16;
    params1.C_dstate_stride = 17;
    params1.C_group_stride = 18;
    params1.u_batch_stride = 19;
    params1.u_d_stride = 20;
    params1.delta_batch_stride = 21;
    params1.delta_d_stride = 22;
    params1.out_batch_stride = 23;
    params1.out_d_stride = 24;
    params1.A_ptr = nullptr;
    params1.B_ptr = nullptr;
    params1.C_ptr = nullptr;
    params1.D_ptr = nullptr;
    params1.u_ptr = nullptr;
    params1.delta_ptr = nullptr;
    params1.delta_bias_ptr = nullptr;
    params1.out_ptr = nullptr;
    params1.x_ptr = nullptr;

    assert(params1.batch == 2);
    assert(params1.dim == 3);
    assert(params1.seqlen == 4);
    assert(params1.dstate == 5);
    assert(params1.n_groups == 6);
    assert(params1.n_chunks == 7);
    assert(params1.dim_ngroups_ratio == 8);
    assert(params1.delta_softplus == true);
    assert(params1.A_d_stride == 9);
    assert(params1.A_dstate_stride == 10);
    assert(params1.B_batch_stride == 11);
    assert(params1.B_d_stride == 12);
    assert(params1.B_dstate_stride == 13);
    assert(params1.B_group_stride == 14);
    assert(params1.C_batch_stride == 15);
    assert(params1.C_d_stride == 16);
    assert(params1.C_dstate_stride == 17);
    assert(params1.C_group_stride == 18);
    assert(params1.u_batch_stride == 19);
    assert(params1.u_d_stride == 20);
    assert(params1.delta_batch_stride == 21);
    assert(params1.delta_d_stride == 22);
    assert(params1.out_batch_stride == 23);
    assert(params1.out_d_stride == 24);

    // Negative case
    SSMParamsBase params2;
    params2.batch = -1;

    try {
        assert(false); // Should not reach here
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
}

// Test case for SSMParamsBwd
void test_SSMParamsBwd() {
    // Happy path
    SSMParamsBwd params1;
    params1.batch = 2;
    params1.dim = 3;
    params1.seqlen = 4;
    params1.dstate = 5;
    params1.n_groups = 6;
    params1.n_chunks = 7;
    params1.dim_ngroups_ratio = 8;
    params1.delta_softplus = true;
    params1.A_d_stride = 9;
    params1.A_dstate_stride = 10;
    params1.B_batch_stride = 11;
    params1.B_d_stride = 12;
    params1.B_dstate_stride = 13;
    params1.B_group_stride = 14;
    params1.C_batch_stride = 15;
    params1.C_d_stride = 16;
    params1.C_dstate_stride = 17;
    params1.C_group_stride = 18;
    params1.u_batch_stride = 19;
    params1.u_d_stride = 20;
    params1.delta_batch_stride = 21;
    params1.delta_d_stride = 22;
    params1.out_batch_stride = 23;
    params1.out_d_stride = 24;
    params1.A_ptr = nullptr;
    params1.B_ptr = nullptr;
    params1.C_ptr = nullptr;
    params1.D_ptr = nullptr;
    params1.u_ptr = nullptr;
    params1.delta_ptr = nullptr;
    params1.delta_bias_ptr = nullptr;
    params1.out_ptr = nullptr;
    params1.x_ptr = nullptr;
    params1.dout_batch_stride = 25;
    params1.dout_d_stride = 26;

    assert(params1.batch == 2);
    assert(params1.dim == 3);
    assert(params1.seqlen == 4);
    assert(params1.dstate == 5);
    assert(params1.n_groups == 6);
    assert(params1.n_chunks == 7);
    assert(params1.dim_ngroups_ratio == 8);
    assert(params1.delta_softplus == true);
    assert(params1.A_d_stride == 9);
    assert(params1.A_dstate_stride == 10);
    assert(params1.B_batch_stride == 11);
    assert(params1.B_d_stride == 12);
    assert(params1.B_dstate_stride == 13);
    assert(params1.B_group_stride == 14);
    assert(params1.C_batch_stride == 15);
    assert(params1.C_d_stride == 16);
    assert(params1.C_dstate_stride == 17);
    assert(params1.C_group_stride == 18);
    assert(params1.u_batch_stride == 19);
    assert(params1.u_d_stride == 20);
    assert(params1.delta_batch_stride == 21);
    assert(params1.delta_d_stride == 22);
    assert(params1.out_batch_stride == 23);
    assert(params1.out_d_stride == 24);
    assert(params1.dout_batch_stride == 25);
    assert(params1.dout_d_stride == 26);

    // Negative case
    SSMParamsBwd params2;
    params2.batch = -1;

    try {
        assert(false); // Should not reach here
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
}

int main() {
    test_SSMScanParamsBase();
    test_SSMParamsBase();
    test_SSMParamsBwd();

    return 0;
}
