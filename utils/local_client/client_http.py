#!/usr/bin/env python
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import sys
import gevent.ssl

import cv2
from torchvision.utils import draw_bounding_boxes

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


def test_infer(model_name,
               input0_data,
               headers=None,
               request_compression_algorithm=None,
               response_compression_algorithm=None):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('INPUT__0', [1, 3, 320, 320], "FP32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)

    outputs.append(httpclient.InferRequestedOutput('boxes__0', binary_data=False))
    outputs.append(httpclient.InferRequestedOutput('scores__1',
                                                   binary_data=False))
    outputs.append(httpclient.InferRequestedOutput('labels__2',
                                                   binary_data=False))

    query_params = {'test_1': 1, 'test_2': 2}
    results = triton_client.infer(
        model_name,
        inputs,
        outputs=outputs,
        query_params=query_params,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm)

    return results


def test_infer_no_outputs(model_name,
                          input0_data,
                          input1_data,
                          headers=None,
                          request_compression_algorithm=None,
                          response_compression_algorithm=None):
    inputs = []
    inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
    inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)
    inputs[1].set_data_from_numpy(input1_data, binary_data=True)

    query_params = {'test_1': 1, 'test_2': 2}
    results = triton_client.infer(
        model_name,
        inputs,
        outputs=None,
        query_params=query_params,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm)

    return results

def draw_bboxes_on_image(image, bboxes, **kwargs):
    color = (255, 0, 0)
    thickness = 5

    for bbox in bboxes:
        start = bbox[0:2]
        end = bbox[2:]
        image = cv2.rectangle(image, start, end, color, thickness)
    return image.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable encrypted link to the server using HTTPS')
    parser.add_argument(
        '-H',
        dest='http_headers',
        metavar="HTTP_HEADER",
        required=False,
        action='append',
        help='HTTP headers to add to inference server requests. ' +
        'Format is -H"Header:Value".')
    parser.add_argument(
        '--request-compression-algorithm',
        type=str,
        required=False,
        default=None,
        help=
        'The compression algorithm to be used when sending request body to server. Default is None.'
    )
    parser.add_argument(
        '--response-compression-algorithm',
        type=str,
        required=False,
        default=None,
        help=
        'The compression algorithm to be used when receiving response body from server. Default is None.'
    )

    FLAGS = parser.parse_args()
    try:
        if FLAGS.ssl:
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url,
                verbose=FLAGS.verbose,
                ssl=True,
                ssl_context_factory=gevent.ssl._create_unverified_context,
                insecure=True)
        else:
            triton_client = httpclient.InferenceServerClient(
                url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "ssd_mobilenet_v3"

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.

    ##TODO: CHANGE THIS TO LOAD FILE
    in_size = 320
    down_points = (in_size, in_size)
    # input0_data = np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size))
    # input0_data = np.array(input0_data, dtype=np.float32)
    # print(input0_data.dtype)

    input_image = cv2.imread("../../data/cars_on_road.jpg")
    input_image = cv2.resize(input_image, down_points, interpolation= cv2.INTER_LINEAR)
    input0_data = np.array(input_image, dtype=np.float32)
    input0_data = np.transpose(input0_data, (2,0,1))
    input0_data = np.expand_dims(input0_data, axis=0)



    if FLAGS.http_headers is not None:
        headers_dict = {
            l.split(':')[0]: l.split(':')[1] for l in FLAGS.http_headers
        }
    else:
        headers_dict = None

    # Infer with requested Outputs
    results = test_infer(model_name, input0_data, headers_dict,
                         FLAGS.request_compression_algorithm,
                         FLAGS.response_compression_algorithm)
    print(results.get_response())

    statistics = triton_client.get_inference_statistics(model_name=model_name,
                                                        headers=headers_dict)
    print(statistics)
    if len(statistics['model_stats']) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)

    # Validate the results by comparing with precomputed values.
    boxes = results.as_numpy('boxes__0')
    scores = results.as_numpy('scores__1')
    labels = results.as_numpy('labels__2')

    print(results.get_response())
    print(f"BBOXES: {boxes}")
    print(f"SCORES: {scores}")
    print(f"LABELS: {labels}")

    out_image = draw_bboxes_on_image(input_image, boxes,)
    status = cv2.imwrite("../../data/cars_on_road_result.jpg", out_image)
    print("Image written to file-system : ", status)

