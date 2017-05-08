function id = convert_joint_order_to_Tompson_order(joint_id)
    % Convert the joint id to MPII joint order
    order_to_MPI = [8 7 11 10 9 12 13 14 3 2 1 4 5 6];
    id = order_to_MPI(joint_id);
end