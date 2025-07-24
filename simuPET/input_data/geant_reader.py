from simuPET import array_lib as np
import itertools as iter


def keep_only_opposite_blocks(hold):
    return (hold[0, 0, -1] - hold[1, 0, -1]) % 2 != 0


def handle_close_multidetections(i, j, hold):
    # hold[:,j]= np.max(hold)
    return 1  # break


def handle_far_multidetections(i, j, hold):
    # hold[:,j]= np.max(hold)
    return 1  # break


def max_number(file_name, offset=400):
    # search backwards for the highest detection number
    import os

    with open(file_name, "rb") as f:
        f.seek(-offset, 2)  # start from end-offset and proceed onwards
        for line in f:
            if line.decode("utf-8").startswith("#"):
                sp = line.split()
    return int(sp[1])


block_dict = {"a": 0, "b": 2, "c": 1, "d": 3}


def read_geant_file(
    geant_file, atol=0.3, max_detections=None, auto_label=True, keep_opposites_only=True
):

    n_values_per_line = 5
    n_blocks = 2
    dont_keep = 2  # data of the line that we dont keep (take in account the block id is moved to last place)
    holding_buffer_size = 12

    if max_detections is None:
        max_detections = max_number(geant_file)

    hold = np.zeros(
        (n_blocks, holding_buffer_size, n_values_per_line)
    )  # 2 detections, 9 buffer
    output = np.zeros((n_blocks, max_detections, n_values_per_line - dont_keep))

    n_total_events = 0
    n_kept_events = 0
    n_close_events = 0

    with open(geant_file, "r") as f:

        for delimiter, event in iter.groupby(f, lambda line: line.startswith("#")):
            if not delimiter:
                try:
                    n_total_events += 1
                    discard = 0  # False
                    # i counts blocks in events, j counts photons per block in event
                    for i, (block, detections) in enumerate(
                        iter.groupby(event, lambda line: line[0])
                    ):  # groupby: need to be sorted
                        if i > 1:
                            break  # too many blocks involved (more than two)

                        detections_far = 0
                        for j, line in enumerate(detections):
                            hold[i, j] = [float(x) for x in line.split()[1:]] + [
                                block_dict[line[0]]
                            ]  # split line into floats

                            if not np.allclose(
                                hold[i, 0][:3], hold[i, j][:3], atol=atol
                            ):
                                detections_far += 1

                        if j > 0:  # more than one detection in one block
                            if not detections_far:  # they look close enough
                                n_close_events += 1
                                discard += handle_close_multidetections(
                                    i, j, hold
                                )  # mark and continue through i
                                # if nothing is done instead, then first detection is taken in hold[:,0,:-dont_keep]

                            else:
                                discard += handle_far_multidetections(
                                    i, j, hold
                                )  # mark as continue through i

                    if (
                        i > 1
                        or discard > 0
                        or (keep_only_opposite_blocks(hold) and keep_opposites_only)
                    ):
                        continue  # too many blocks involved or strange stuff

                    else:
                        output[:, n_kept_events, :] = hold[:, 0, :-dont_keep]
                        n_kept_events += 1
                except:
                    print(event)

    return output[:, :n_kept_events, :]
