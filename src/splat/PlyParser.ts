import { SplatPointCloud } from './types.js';

export class PlyParser {
  /**
   * Sigmoid activation for spherical harmonics and opacity decoding
   */
  public static sigmoid(v: number): number {
    return 1 / (1 + Math.exp(-v));
  }

  /**
   * Computes the 3D covariance matrix elements for each Gaussian based on its scale and rotation quaternion.
   * Covariance Sigma = R * S * S^T * R^T
   * Returns packed upper triangular components: [cov11, cov12, cov13, cov22, cov23, cov33] per splat.
   */
  public static computeCovariances(
    scales: Float32Array,
    rotations: Float32Array,
    count: number
  ): Float32Array {
    const covariances = new Float32Array(count * 6);

    for (let i = 0; i < count; i++) {
      const si = i * 3;
      const ri = i * 4;
      const ci = i * 6;

      const sx = scales[si] ?? 0;
      const sy = scales[si + 1] ?? 0;
      const sz = scales[si + 2] ?? 0;

      const qx = rotations[ri] ?? 0;
      const qy = rotations[ri + 1] ?? 0;
      const qz = rotations[ri + 2] ?? 0;
      const qw = rotations[ri + 3] ?? 1;

      // Rotation matrix from normalized quaternion
      const r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
      const r01 = 2.0 * (qx * qy - qz * qw);
      const r02 = 2.0 * (qx * qz + qy * qw);

      const r10 = 2.0 * (qx * qy + qz * qw);
      const r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
      const r12 = 2.0 * (qy * qz - qx * qw);

      const r20 = 2.0 * (qx * qz - qy * qw);
      const r21 = 2.0 * (qy * qz + qx * qw);
      const r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

      // M = R * S
      const m00 = r00 * sx;
      const m01 = r01 * sy;
      const m02 = r02 * sz;

      const m10 = r10 * sx;
      const m11 = r11 * sy;
      const m12 = r12 * sz;

      const m20 = r20 * sx;
      const m21 = r21 * sy;
      const m22 = r22 * sz;

      // Sigma = M * M^T
      // cov00
      covariances[ci] = m00 * m00 + m01 * m01 + m02 * m02;
      // cov01
      covariances[ci + 1] = m00 * m10 + m01 * m11 + m02 * m12;
      // cov02
      covariances[ci + 2] = m00 * m20 + m01 * m21 + m02 * m22;
      // cov11
      covariances[ci + 3] = m10 * m10 + m11 * m11 + m12 * m12;
      // cov12
      covariances[ci + 4] = m10 * m20 + m11 * m21 + m12 * m22;
      // cov22
      covariances[ci + 5] = m20 * m20 + m21 * m21 + m22 * m22;
    }

    return covariances;
  }

  /**
   * Parse Stanford PLY binary little-endian format from Gaussian Splatting trained models.
   */
  public static parsePly(buffer: ArrayBuffer): SplatPointCloud {
    const textDecoder = new TextDecoder();
    const headerBytes = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 4096));
    const headerText = textDecoder.decode(headerBytes);

    const endHeaderIndex = headerText.indexOf('end_header\n');
    if (endHeaderIndex === -1) {
      throw new Error('Invalid PLY file: end_header not found in header preamble');
    }

    const header = headerText.substring(0, endHeaderIndex + 'end_header\n'.length);
    const lines = header.split(/\r?\n/);

    const formatLine = lines.find((line) => line.trim().startsWith('format '))?.trim();
    if (formatLine !== 'format binary_little_endian 1.0') {
      throw new Error('Unsupported PLY format: expected binary_little_endian 1.0');
    }

    if (lines[0]?.trim() !== 'ply') {
      throw new Error('Invalid PLY file: missing "ply" magic header');
    }

    let vertexCount = 0;
    const propertyTypes: Array<{ name: string; type: string; byteSize: number }> = [];

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (line.startsWith('element vertex')) {
        const parts = line.split(/\s+/);
        const countStr = parts[2];
        if (countStr) {
          vertexCount = parseInt(countStr, 10);
        }
      } else if (line.startsWith('property')) {
        const parts = line.split(/\s+/);
        const type = parts[1] ?? 'float';
        const name = parts[2] ?? '';

        const byteSizes: Record<string, number> = {
          float: 4, float32: 4, int: 4, int32: 4, uint: 4, uint32: 4,
          double: 8, float64: 8,
          uchar: 1, uint8: 1, char: 1, int8: 1,
          short: 2, int16: 2, ushort: 2, uint16: 2
        };
        const byteSize = byteSizes[type];
        if (byteSize === undefined) {
          throw new Error(`Unsupported PLY property type: ${type}`);
        }
        propertyTypes.push({ name, type, byteSize });
      }
    }

    if (!Number.isSafeInteger(vertexCount) || vertexCount <= 0) {
      throw new Error('Invalid PLY file: vertex count is 0 or unparsed');
    }

    const headerByteLength = new TextEncoder().encode(header).length;
    let byteStride = 0;
    for (const prop of propertyTypes) {
      byteStride += prop.byteSize;
    }
    const payloadBytes = vertexCount * byteStride;
    if (byteStride <= 0 || !Number.isSafeInteger(payloadBytes)
        || payloadBytes > buffer.byteLength - headerByteLength) {
      throw new Error('Invalid PLY file: vertex data is truncated or has an invalid stride');
    }

    const positions = new Float32Array(vertexCount * 3);
    const scales = new Float32Array(vertexCount * 3);
    const rotations = new Float32Array(vertexCount * 4);
    const colors = new Float32Array(vertexCount * 4);

    const view = new DataView(buffer, headerByteLength);

    // Constant for SH degree 0
    const SH_C0 = 0.28209479177387814;

    for (let i = 0; i < vertexCount; i++) {
      const rowOffset = i * byteStride;
      let offset = rowOffset;

      let x = 0;
      let y = 0;
      let z = 0;
      let scale0 = 0;
      let scale1 = 0;
      let scale2 = 0;
      let rot0 = 1; // w
      let rot1 = 0; // x
      let rot2 = 0; // y
      let rot3 = 0; // z
      let opacity = 0;
      let f_dc_0 = 0;
      let f_dc_1 = 0;
      let f_dc_2 = 0;
      let red = -1;
      let green = -1;
      let blue = -1;

      for (const prop of propertyTypes) {
        let val = 0;
        switch (prop.type) {
          case 'float':
          case 'float32':
            val = view.getFloat32(offset, true);
            break;
          case 'double':
          case 'float64':
            val = view.getFloat64(offset, true);
            break;
          case 'char':
          case 'int8':
            val = view.getInt8(offset);
            break;
          case 'uchar':
          case 'uint8':
            val = view.getUint8(offset);
            break;
          case 'short':
          case 'int16':
            val = view.getInt16(offset, true);
            break;
          case 'ushort':
          case 'uint16':
            val = view.getUint16(offset, true);
            break;
          case 'int':
          case 'int32':
            val = view.getInt32(offset, true);
            break;
          case 'uint':
          case 'uint32':
            val = view.getUint32(offset, true);
            break;
        }

        switch (prop.name) {
          case 'x': x = val; break;
          case 'y': y = val; break;
          case 'z': z = val; break;
          case 'scale_0': scale0 = Math.exp(val); break;
          case 'scale_1': scale1 = Math.exp(val); break;
          case 'scale_2': scale2 = Math.exp(val); break;
          case 'rot_0': rot0 = val; break; // qw
          case 'rot_1': rot1 = val; break; // qx
          case 'rot_2': rot2 = val; break; // qy
          case 'rot_3': rot3 = val; break; // qz
          case 'opacity': opacity = PlyParser.sigmoid(val); break;
          case 'f_dc_0': f_dc_0 = val; break;
          case 'f_dc_1': f_dc_1 = val; break;
          case 'f_dc_2': f_dc_2 = val; break;
          case 'red': red = val / 255.0; break;
          case 'green': green = val / 255.0; break;
          case 'blue': blue = val / 255.0; break;
        }

        offset += prop.byteSize;
      }

      // If direct rgb was not provided, calculate from Spherical Harmonics DC component
      let r = red;
      let g = green;
      let b = blue;
      if (r < 0) {
        r = Math.min(1.0, Math.max(0.0, 0.5 + SH_C0 * f_dc_0));
        g = Math.min(1.0, Math.max(0.0, 0.5 + SH_C0 * f_dc_1));
        b = Math.min(1.0, Math.max(0.0, 0.5 + SH_C0 * f_dc_2));
      }

      // Normalize quaternion
      const qlen = Math.hypot(rot0, rot1, rot2, rot3) || 1.0;
      const qw = rot0 / qlen;
      const qx = rot1 / qlen;
      const qy = rot2 / qlen;
      const qz = rot3 / qlen;

      const idx3 = i * 3;
      const idx4 = i * 4;

      positions[idx3] = x;
      positions[idx3 + 1] = y;
      positions[idx3 + 2] = z;

      scales[idx3] = scale0;
      scales[idx3 + 1] = scale1;
      scales[idx3 + 2] = scale2;

      rotations[idx4] = qx;
      rotations[idx4 + 1] = qy;
      rotations[idx4 + 2] = qz;
      rotations[idx4 + 3] = qw;

      colors[idx4] = r;
      colors[idx4 + 1] = g;
      colors[idx4 + 2] = b;
      colors[idx4 + 3] = opacity;
    }

    const covariances = PlyParser.computeCovariances(scales, rotations, vertexCount);

    return {
      count: vertexCount,
      positions,
      scales,
      rotations,
      colors,
      covariances
    };
  }

  /**
   * Serializes a SplatPointCloud to binary PLY format for caching and export.
   */
  public static serializePly(cloud: SplatPointCloud): ArrayBuffer {
    const count = cloud.count;
    const headerString = 
      `ply\n` +
      `format binary_little_endian 1.0\n` +
      `element vertex ${count}\n` +
      `property float x\n` +
      `property float y\n` +
      `property float z\n` +
      `property float scale_0\n` +
      `property float scale_1\n` +
      `property float scale_2\n` +
      `property float rot_0\n` +
      `property float rot_1\n` +
      `property float rot_2\n` +
      `property float rot_3\n` +
      `property float opacity\n` +
      `property float f_dc_0\n` +
      `property float f_dc_1\n` +
      `property float f_dc_2\n` +
      `end_header\n`;

    const headerBytes = new TextEncoder().encode(headerString);
    const vertexByteSize = 14 * 4; // 14 floats (3 pos + 3 scale + 4 rot + 1 opacity + 3 sh)
    const totalBytes = headerBytes.byteLength + count * vertexByteSize;

    const buffer = new ArrayBuffer(totalBytes);
    const uint8View = new Uint8Array(buffer);
    uint8View.set(headerBytes, 0);

    const dataView = new DataView(buffer, headerBytes.byteLength);
    const SH_C0 = 0.28209479177387814;

    for (let i = 0; i < count; i++) {
      const offset = i * vertexByteSize;
      const i3 = i * 3;
      const i4 = i * 4;

      const px = cloud.positions[i3] ?? 0;
      const py = cloud.positions[i3 + 1] ?? 0;
      const pz = cloud.positions[i3 + 2] ?? 0;

      const sx = Math.log(Math.max(1e-6, cloud.scales[i3] ?? 1e-4));
      const sy = Math.log(Math.max(1e-6, cloud.scales[i3 + 1] ?? 1e-4));
      const sz = Math.log(Math.max(1e-6, cloud.scales[i3 + 2] ?? 1e-4));

      const qx = cloud.rotations[i4] ?? 0;
      const qy = cloud.rotations[i4 + 1] ?? 0;
      const qz = cloud.rotations[i4 + 2] ?? 0;
      const qw = cloud.rotations[i4 + 3] ?? 1;

      const r = cloud.colors[i4] ?? 1;
      const g = cloud.colors[i4 + 1] ?? 1;
      const b = cloud.colors[i4 + 2] ?? 1;
      const opacity = cloud.colors[i4 + 3] ?? 1;

      // Inverse sigmoid for opacity: log(p / (1 - p))
      const pClamped = Math.min(0.9999, Math.max(0.0001, opacity));
      const rawOpacity = Math.log(pClamped / (1 - pClamped));

      const f_dc_0 = (r - 0.5) / SH_C0;
      const f_dc_1 = (g - 0.5) / SH_C0;
      const f_dc_2 = (b - 0.5) / SH_C0;

      dataView.setFloat32(offset, px, true);
      dataView.setFloat32(offset + 4, py, true);
      dataView.setFloat32(offset + 8, pz, true);

      dataView.setFloat32(offset + 12, sx, true);
      dataView.setFloat32(offset + 16, sy, true);
      dataView.setFloat32(offset + 20, sz, true);

      dataView.setFloat32(offset + 24, qw, true);
      dataView.setFloat32(offset + 28, qx, true);
      dataView.setFloat32(offset + 32, qy, true);
      dataView.setFloat32(offset + 36, qz, true);

      dataView.setFloat32(offset + 40, rawOpacity, true);
      dataView.setFloat32(offset + 44, f_dc_0, true);
      dataView.setFloat32(offset + 48, f_dc_1, true);
      dataView.setFloat32(offset + 52, f_dc_2, true);
    }

    return buffer;
  }
}
