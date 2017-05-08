function id = convert_joint_order(joint_id)
% Convert the joint id to LSP joint order
    order_to_LSP = [14 13 9 8 7 10 11 12 3 2 1 4 5 6];
    id = order_to_LSP(joint_id);
end