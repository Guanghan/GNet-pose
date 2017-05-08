function id = get_official_id(joint_id)
    % Convert the joint id to MPII joint order to official id.
    % my joint: 1-14
    % official: 0-15
    order_to_MPI_official = [9 8 12 11 10 13 14 15 2 1 0 3 4 5];
    id = order_to_MPI_official(joint_id);
end
