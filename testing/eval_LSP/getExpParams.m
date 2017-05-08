function p = getExpParams(predidx)

switch predidx
    case 0
        p.name = 'Pishchulin et., CVPR''13';
        p.predFilename = '';
    case 1
        p.name = 'Pishchulin et., ICCV''13';
        p.predFilename = './pred/pishchulin13iccv/pred_keypoints_lsp_oc';
        p.colorIdxs = [1 1];
    case 2
        p.name = 'Tompson et al., NIPS''14';
        p.predFilename = './pred/tompson14nips/pred_keypoints_lsp_pc';
        p.colorIdxs = [2 1];
    case 3
        p.name = 'Chen&Yuille, NIPS''14';
        p.predFilename = './pred/chen14nips/pred_keypoints_lsp_oc';
        p.colorIdxs = [5 1];
    case 4
        p.name = 'Ramakrishna et al., ECCV''14';
        p.predFilename = './pred/ramakrishna14eccv/pred_keypoints_lsp_oc';
        p.colorIdxs = [4 1];
    case 5
        p.name = 'Ouyang et al., CVPR''14';
        p.predFilename = './pred/ouyang14cvpr/pred_keypoints_lsp_oc';
        p.colorIdxs = [7 1];
    case 6
        p.name = 'Pishchulin et., ICCV''13';
        p.predFilename = './pred/pishchulin13iccv/pred_keypoints_lsp_pc';
        p.colorIdxs = [1 1];
    case 7
        p.name = 'Pishchulin et., ICCV''13';
        p.predFilename = './pred/pishchulin13iccv/pred_sticks_lsp_oc';
        p.colorIdxs = [1 1];
    case 8
        p.name = 'Pishchulin et., ICCV''13';
        p.predFilename = './pred/pishchulin13iccv/pred_sticks_lsp_pc';
        p.colorIdxs = [1 1];
    case 9
        p.name = 'Tompson et al., NIPS''14';
        p.predFilename = './pred/tompson14nips/pred_sticks_lsp_pc';
        p.colorIdxs = [2 1];
    case 10
        p.name = 'Ramakrishna et al., ECCV''14';
        p.predFilename = './pred/ramakrishna14eccv/pred_sticks_lsp_oc';
        p.colorIdxs = [4 1];
    case 11
        p.name = 'Chen&Yuille, NIPS''14';
        p.predFilename = './pred/chen14nips/pred_sticks_lsp_oc';
        p.colorIdxs = [5 1];
    case 12
        p.name = 'Ouyang et al., CVPR''14';
        p.predFilename = './pred/ouyang14cvpr/pred_sticks_lsp_oc';
        p.colorIdxs = [7 1];
    case 13
        p.name = 'Pishchulin et al., CVPR''13';
        p.predFilename = './pred/pishchulin13cvpr/pred_sticks_lsp_oc';
        p.colorIdxs = [8 1];
    case 14
        p.name = 'Kiefel&Gehler, ECCV''14';
        p.predFilename = './pred/kiefel14eccv/pred_sticks_lsp_oc';
        p.colorIdxs = [8 2];
    case 15
        p.name = 'Kiefel&Gehler, ECCV''14';
        p.predFilename = './pred/kiefel14eccv/pred_keypoints_lsp_oc';
        p.colorIdxs = [8 3];  
   
    case 19
        p.name = 'Bulat et al., ECCV''16';
        p.predFilename = 'pred/1st/pred_keypoints_lsp_pc';
        p.colorIdxs = [1 1];
        case 20
        p.name = 'Wei el al., CVPR''16';
        p.predFilename = 'pred/2nd/pred_keypoints_lsp_pc';
        p.colorIdxs = [2 1];
          case 21
        p.name = 'Insafutdinov et al., ECCV''16   .';
        p.predFilename = 'pred/3rd/pred_keypoints_lsp_pc';
        p.colorIdxs = [3 1];
          case 22
        p.name = 'Pishchulin et al., CVPR''16';
        p.predFilename = 'pred/4th/pred_keypoints_lsp_pc';
        p.colorIdxs = [4 1];
          case 23
        p.name = 'Lifshitz et al., ECCV''16';
        p.predFilename = 'pred/5th/pred_keypoints_lsp_pc';
        p.colorIdxs = [5 1];
          case 24
        p.name = 'Belagiannis et al., FG''17';
        p.predFilename = 'pred/6th/pred_keypoints_lsp_pc';
        p.colorIdxs = [6 1];
          case 25
        p.name = 'Yu et al., ECCV''16';
        p.predFilename = 'pred/7th/pred_keypoints_lsp_pc';
        p.colorIdxs = [7 1];
          case 26
        p.name = 'Rafi et al., BMVC''16';  
        p.predFilename = 'pred/8th/pred_keypoints_lsp_pc';
        p.colorIdxs = [1 2];
          case 27
        p.name = 'Yang et al., CVPR''16'; 
        p.predFilename = 'pred/9th/pred_keypoints_lsp_pc';
        p.colorIdxs = [2 2];
          case 28
        p.name = 'Chen&Yuille, NIPS''14';  
        p.predFilename = 'pred/10th/pred_keypoints_lsp_pc';
        p.colorIdxs = [3 2];
          case 29
        p.name = 'Fan et al., CVPR''15';  
        p.predFilename = 'pred/12th/pred_keypoints_lsp_pc';
        p.colorIdxs = [4 2];
          case 30
        p.name = 'Tompson et al., NIPS''14';  
        p.predFilename = 'pred/13th/pred_keypoints_lsp_pc';
        p.colorIdxs = [4 3];
          case 31
        p.name = 'Pishchulin et al., ICCV''13';  
        p.predFilename = 'pred/14th/pred_keypoints_lsp_pc';
        p.colorIdxs = [4 4];
          case 32
        p.name = 'Wang&Li, CVPR''13';  
        p.predFilename = 'pred/15th/pred_keypoints_lsp_pc';
        p.colorIdxs = [2 3];
       case 33  % Add GNet. Guanghan Ning. Dec 2016
        p.name = 'Ours';
        p.predFilename = 'pred/ning17iccv/pred_keypoints_lsp';
        p.colorIdxs = [8 1];
          
        
end

p.colorName = getColor(p.colorIdxs);
p.colorName = p.colorName ./ 255;

end