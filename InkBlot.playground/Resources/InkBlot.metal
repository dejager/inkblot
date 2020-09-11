#include <metal_stdlib>
using namespace metal;

float3 random3(float3 color) {
  float i = 4096.0 * sin(dot(color, float3(17.0, 59.4, 15.0)));
  float3 r;
  r.z = fract(512.0 * i);
  i *= .125;
  r.x = fract(512.0 * i);
  i *= .125;
  r.y = fract(512.0 * i);
  return r - 0.5;
}

float simplex3d(float3 p) {

  // simplex noise
  float F3 =  0.3333333;
  float G3 =  0.1666667;

  float3 s = floor(p + dot(p, float3(F3)));
  float3 x = p - s + dot(s, float3(G3));

  float3 e = step(float3(0.0), x - x.yzx);
  float3 i1 = e*(1.0 - e.zxy);
  float3 i2 = 1.0 - e.zxy*(1.0 - e);

  float3 x1 = x - i1 + G3;
  float3 x2 = x - i2 + 2.0 * G3;
  float3 x3 = x - 1.0 + 3.0 * G3;

  float4 w, d;

  w.x = dot(x, x);
  w.y = dot(x1, x1);
  w.z = dot(x2, x2);
  w.w = dot(x3, x3);

  w = max(0.6 - w, 0.0);

  d.x = dot(random3(s), x);
  d.y = dot(random3(s + i1), x1);
  d.z = dot(random3(s + i2), x2);
  d.w = dot(random3(s + 1.0), x3);

  w *= w;
  w *= w;
  d *= w;

  return dot(d, float4(52.0));
}

float fbm(float3 p)
{
  float f = 0.0;
  float frequency = 1.0;
  float amplitude = 0.5;
  for (int i = 0; i < 5; i++) {
    f += simplex3d(p * frequency) * amplitude;
    amplitude *= 0.5;
    frequency *= 2.0 + float(i) / 100.0;
  }
  return min(f, 1.0);
}


kernel void rorschach(texture2d<float, access::write> o[[texture(0)]],
                      constant float &time [[buffer(0)]],
                      constant float2 *touchEvent [[buffer(1)]],
                      constant int &numberOfTouches [[buffer(2)]],
                      ushort2 gid [[thread_position_in_grid]]) {

  // config
  float3 ink = float3(0.01, 0.01, 0.1);
  float3 paper = float3(1.0);

  float speed = 0.0075;
  float shadeContrast = 0.55;

  // coordinates
  int width = o.get_width();
  int height = o.get_height();
  float2 res = float2(width, height);

  float2 uv = float2(gid) / res;

  float2 coord = 1.0 - uv * 2.0;
  uv.x = 1.0 - abs(1.0 - uv.x * 2.0);

  float3 p = float3(uv, time * speed);

  // sample noise
  float blot = fbm(p * 3.0 + 8.0);
  float shade = fbm(p * 2.0 + 16.0);

  // threshold
  blot = (blot + (sqrt(uv.x) - abs(0.5 - uv.y)));
  blot = smoothstep(0.65, 0.71, blot) * max(1.0 - shade * shadeContrast, 0.0);

  // color
  float4 color = float4(mix(paper, ink, blot), 1.0);

  o.write(color, gid);
}
