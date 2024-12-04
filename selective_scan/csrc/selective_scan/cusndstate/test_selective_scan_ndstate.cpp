#include <iostream>
#include <cassert>

// Forward declarations of the functions to be tested
void ssm_scan_params_base(SSMScanParamsBase &params);
void ssm_params_base(SSMParamsBase &params);
void ssm_params_bwd(SSMParamsBwd &params);

// Test cases for SSMScanParamsBase
void test_ssm_scan_params_base() {
    SSMScanParamsBase params;
    params.batch = 2;
    params.seqlen = 3;
    params.n_chunks = 4;
    params.a_batch_stride = 5;
    params.b_batch_stride = 6;
    params.out_batch_stride = 7;
    params.a_ptr = nullptr;
    params.b_ptr = nullptr;
    params.out_ptr = nullptr;
    params.x_ptr = nullptr;

    ssm_scan_params_base(params);

    assert(params.batch == 2);
    assert(params.seqlen == 3);
    assert(params.n_chunks == 4);
    assert(params.a_batch_stride == 5);
    assert(params.b_batch_stride == 6);
    assert(params.out_batch_stride == 7);
    assert(params.a_ptr != nullptr);
    assert(params.b_ptr != nullptr);
    assert(params.out_ptr != nullptr);
    assert(params.x_ptr != nullptr);
}

// Test cases for SSMParamsBase
void test_ssm_params_base() {
    SSMParamsBase params;
    params.batch = 2;
    params.dim = 3;
    params.seqlen = 4;
    params.n_groups = 5;
    params.n_chunks = 6;
    params.dim_ngroups_ratio = 7;
    params.delta_softplus = true;
    params.A_d_stride = 8;
    params.B_batch_stride = 9;
    params.B_d_stride = 10;
    params.B_group_stride = 11;
    params.C_batch_stride = 12;
    params.C_d_stride = 13;
    params.C_group_stride = 14;
    params.u_batch_stride = 15;
    params.u_d_stride = 16;
    params.delta_batch_stride = 17;
    params.delta_d_stride = 18;
    params.out_batch_stride = 19;
    params.out_d_stride = 20;
    params.A_ptr = nullptr;
    params.B_ptr = nullptr;
    params.C_ptr = nullptr;
    params.D_ptr = nullptr;
    params.u_ptr = nullptr;
    params.delta_ptr = nullptr;
    params.delta_bias_ptr = nullptr;
    params.out_ptr = nullptr;
    params.x_ptr = nullptr;

    ssm_params_base(params);

    assert(params.batch == 2);
    assert(params.dim == 3);
    assert(params.seqlen == 4);
    assert(params.n_groups == 5);
    assert(params.n_chunks == 6);
    assert(params.dim_ngroups_ratio == 7);
    assert(params.delta_softplus == true);
    assert(params.A_d_stride == 8);
    assert(params.B_batch_stride == 9);
    assert(params.B_d_stride == 10);
    assert(params.B_group_stride == 11);
    assert(params.C_batch_stride == 12);
    assert(params.C_d_stride == 13);
    assert(params.C_group_stride == 14);
    assert(params.u_batch_stride == 15);
    assert(params.u_d_stride == 16);
    assert(params.delta_batch_stride == 17);
    assert(params.delta_d_stride == 18);
    assert(params.out_batch_stride == 19);
    assert(params.out_d_stride == 20);
    assert(params.A_ptr != nullptr);
    assert(params.B_ptr != nullptr);
    assert(params.C_ptr != nullptr);
    assert(params.D_ptr != nullptr);
    assert(params.u_ptr != nullptr);
    assert(params.delta_ptr != nullptr);
    assert(params.delta_bias_ptr != nullptr);
    assert(params.out_ptr != nullptr);
    assert(params.x_ptr != nullptr);
}

// Test cases for SSMParamsBwd
void test_ssm_params_bwd() {
    SSMParamsBwd params;
    params.batch = 2;
    params.dim = 3;
    params.seqlen = 4;
    params.n_groups = 5;
    params.n_chunks = 6;
    params.dim_ngroups_ratio = 7;
    params.delta_softplus = true;
    params.A_d_stride = 8;
    params.B_batch_stride = 9;
    params.B_d_stride = 10;
    params.B_group_stride = 11;
    params.C_batch_stride = 12;
    params.C_d_stride = 13;
    params.C_group_stride = 14;
    params.u_batch_stride = 15;
    params.u_d_stride = 16;
    params.delta_batch_stride = 17;
    params.delta_d_stride = 18;
    params.out_batch_stride = 19;
    params.out_d_stride = 20;
    params.dout_batch_stride = 21;
    params.dout_d_stride = 22;
    params.dA_d_stride = 23;
    params.dB_batch_stride = 24;
    params.dB_group_stride = 25;
    params.dB_d_stride = 26;
    params.dC_batch_stride = 27;
    params.dC_group_stride = 28;
    params.dC_d_stride = 29;
    params.du_batch_stride = 30;
    params.du_d_stride = 31;
    params.ddelta_batch_stride = 32;
    params.ddelta_d_stride = 33;
    params.A_ptr = nullptr;
    params.B_ptr = nullptr;
    params.C_ptr = nullptr;
    params.D_ptr = nullptr;
    params.u_ptr = nullptr;
    params.delta_ptr = nullptr;
    params.delta_bias_ptr = nullptr;
    params.out_ptr = nullptr;
    params.x_ptr = nullptr;

    ssm_params_bwd(params);

    assert(params.batch == 2);
    assert(params.dim == 3);
    assert(params.seqlen == 4);
    assert(params.n_groups == 5);
    assert(params.n_chunks == 6);
    assert(params.dim_ngroups_ratio == 7);
    assert(params.delta_softplus == true);
    assert(params.A_d_stride == 8);
    assert(params.B_batch_stride == 9);
    assert(params.B_d_stride == 10);
    assert(params.B_group_stride == 11);
    assert(params.C_batch_stride == 12);
    assert(params.C_d_stride == 13);
    assert(params.C_group_stride == 14);
    assert(params.u_batch_stride == 15);
    assert(params.u_d_stride == 16);
    assert(params.delta_batch_stride == 17);
    assert(params.delta_d_stride == 18);
    assert(params.out_batch_stride == 19);
    assert(params.out_d_stride == 20);
    assert(params.dout_batch_stride == 21);
    assert(params.dout_d_stride == 22);
    assert(params.dA_d_stride == 23);
    assert(params.dB_batch_stride == 24);
    assert(params.dB_group_stride == 25);
    assert(params.dB_d_stride == 26);
    assert(params.dC_batch_stride == 27);
    assert(params.dC_group_stride == 28);
    assert(params.dC_d_stride == 29);
    assert(params.du_batch_stride == 30);
    assert(params.du_d_stride == 31);
    assert(params.ddelta_batch_stride == 32);
    assert(params.ddelta_d_stride == 33);
    assert(params.A_ptr != nullptr);
    assert(params.B_ptr != nullptr);
    assert(params.C_ptr != nullptr);
    assert(params.D_ptr != nullptr);
    assert(params.u_ptr != nullptr);
    assert(params.delta_ptr != nullptr);
    assert(params.delta_bias_ptr != nullptr);
    assert(params.out_ptr != nullptr);
    assert(params.x_ptr != nullptr);
}

int main() {
    test_ssm_scan_params_base();
    test_ssm_params_base();
    test_ssm_params_bwd();

    return 0;
}
