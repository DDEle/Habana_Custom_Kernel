/**********************************************************************
Copyright (c) 2021 Habana Labs.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

template <int width = 0, int height = 1>
void _main(tensor input0, tensor input1, tensor output) {

  const int5 is_start = get_index_space_offset();
  const int5 is_end = get_index_space_size() + is_start;

  [[gnu::unused]] float64 x00, x01;
  [[gnu::unused]] float64 o0;
  [[gnu::unused]] float64 zeros = 0.f;

  int5 targetCoords = {0, 0, 0, 0, 0};

  for (int d5 = is_start[4]; d5 < is_end[4]; d5 += 1) {
    for (int d4 = is_start[3]; d4 < is_end[3]; d4 += 1) {
      for (int d3 = is_start[2]; d3 < is_end[2]; d3 += 1) {
        for (int i = is_start[height]; i < is_end[height]; i += 1) {
          for (int j = is_start[width]; j < is_end[width]; j += 1) {
            targetCoords[4] = d5;
            targetCoords[3] = d4;
            targetCoords[2] = d3;
            targetCoords[1] = i;
            targetCoords[0] = j;
            x00 = v_f32_ld_tnsr_b(targetCoords, input0);
            x01 = v_f32_ld_tnsr_b(targetCoords, input1);

            // o0 = v_f32_add_b(x00, x01);

            // float64 fclass00 = v_f32_fclass_b(x00);
            // float64 fclass01 = v_f32_fclass_b(x01);
            // o0 = v_f32_calc_fp_special_b(fclass00, fclass00, e_fp_recip,
            // zeros);

            // o0 = v_broadcast_element_f32(x00, 3);

            o0 = v_reciprocal_f32(x00);

            v_f32_st_tnsr(targetCoords, output, o0);
          }
        }
      }
    }
  }
}

void main(tensor input0, tensor input1, tensor output) {
  _main<>(input0, input1, output);
}