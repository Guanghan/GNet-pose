function id = convert_joint_order_to_gt_order(joint_id)
    order_to_MPI = [14 13 9 8 7 10 11 12 3 2 1 4 5 6];
    id = order_to_MPI(joint_id);
end
