# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import json
import shutil
import subprocess
import numpy as np
import os.path as osp


def run_openpose(
    video_file, output_folder, docker_image, vis=False,
):
    render = 1 if vis else 0
    display = 2 if vis else 0

    docker_preamble = [
        "docker",
        "run",
        "--gpus",
        "all",
        "--rm",
        "-it",
        "-v",
        video_file + ":" + video_file,
        "-v",
        output_folder + ":" + output_folder,
        docker_image,
    ]

    cmd = [
        "build/examples/openpose/openpose.bin",
        "--model_pose",
        "BODY_21A",
        "--tracking",
        "1",
        "--render_pose",
        str(render),
        "--video",
        video_file,
        "--write_json",
        output_folder,
        "--display",
        str(display),
    ]

    print("Executing", " ".join(docker_preamble + cmd))
    subprocess.call(docker_preamble + cmd)


def read_posetrack_keypoints(output_folder):

    people = dict()

    for idx, result_file in enumerate(sorted(os.listdir(output_folder))):
        json_file = osp.join(output_folder, result_file)
        data = json.load(open(json_file))
        # print(idx, data)
        for person in data["people"]:
            person_id = person["person_id"][0]
            joints2d = person["pose_keypoints_2d"]
            if person_id in people.keys():
                people[person_id]["joints2d"].append(joints2d)
                people[person_id]["frames"].append(idx)
            else:
                people[person_id] = {
                    "joints2d": [],
                    "frames": [],
                }
                people[person_id]["joints2d"].append(joints2d)
                people[person_id]["frames"].append(idx)

    for k in people.keys():
        people[k]["joints2d"] = np.array(people[k]["joints2d"]).reshape(
            (len(people[k]["joints2d"]), -1, 3)
        )
        people[k]["frames"] = np.array(people[k]["frames"])

    return people


def run_posetracker(video_file, staf_folder, posetrack_output_folder="/tmp", display=False):
    posetrack_output_folder = os.path.join(
        posetrack_output_folder, f"{os.path.basename(video_file)}_posetrack"
    )

    # run posetrack on video
    run_openpose(video_file, posetrack_output_folder, vis=display, staf_folder=staf_folder)

    people_dict = read_posetrack_keypoints(posetrack_output_folder)

    # shutil.rmtree(posetrack_output_folder)

    return people_dict
