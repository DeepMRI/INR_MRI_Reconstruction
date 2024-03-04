import os
import data
import utils
import configs
import model
import torch
import numpy as np



if __name__ == '__main__':

    config = configs.config()
    device = config.device
    print('using device: %s.' % device)

    # Create directories for experimental logs
    log_file = open(os.path.join(config.snap_path, 'log.txt'), 'a+')

    # network
    model = model.network(config, scale_embed=True).to(device)
    # loss
    loss_fn_1 = torch.nn.L1Loss().to(device)
    loss_fn_2 = torch.nn.MSELoss().to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2),
                                 weight_decay=config.weight_decay, amsgrad=True)
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=0.5)
    # load checkpoint
    metric_ssim = 0
    start_iter = 0
    checkpoint_dir = os.path.join(config.snap_path, 'checkpoint.pth')
    if config.load_checkpoint:
        if os.path.isfile(checkpoint_dir):
            checkpoint = torch.load(checkpoint_dir, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_iter = checkpoint['iter']
            metric_ssim = checkpoint['ssim']
            print('load checkpoint from {} successfully, restart from iteration {}.'.format(config.snap_path, start_iter + 1))

    # data loader
    train_loader, val_loader = data.get_dataloader(config)

    if config.is_train:
        for step in range(start_iter, config.n_iters):

            model.train()
            optimizer.zero_grad()

            inputs, targets = next(train_loader)
            pred_image = model(inputs['coords'].to(device),
                               inputs['undersampled_image'].to(device),
                               inputs['down_scale'].to(device)).permute(0, 3, 1, 2)
            # image domain loss
            loss_image = loss_fn_1(pred_image, targets['image'].to(device))

            loss_image.backward()
            optimizer.step()

            if (step + 1) % config.log_step == 0:
                print('Step: [%d/%d], lr: [%.8f], loss_image=%.5f'
                    % (step + 1, config.n_iters, optimizer.param_groups[0]['lr'], loss_image.item()))
                log_file.write('Step: [%d/%d], lr: [%.8f], loss_image=%.5f\n'
                    % (step + 1, config.n_iters, optimizer.param_groups[0]['lr'], loss_image.item()))

            if (step + 1) % config.val_step == 0:
                with torch.no_grad():
                    ssim, psnr, image_loss = 0, 0, 0
                    for i, batch in enumerate(val_loader):
                        inputs, targets = batch
                        pred_image = model(inputs['coords'].to(device),
                                           inputs['undersampled_image'].to(device),
                                           inputs['down_scale'].to(device)).permute(0, 3, 1, 2)
                        # image domain loss
                        image_loss += loss_fn_1(pred_image, targets['image'].to(device))
                        ssim += utils.ssim(pred_image, targets['image'].to(device)).item()
                        psnr += utils.psnr(pred_image, targets['image'].to(device)).item()

                    image_loss /= len(val_loader)
                    ssim /= len(val_loader)
                    psnr /= len(val_loader)
                    print('Step: [%d/%d], val_image_loss=%.5f, val_ssim=%.5f, val_psnr=%.5f' % (step + 1, config.n_iters, image_loss, ssim, psnr))
                    log_file.write('Step: [%d/%d], val_image_loss=%.5f, val_ssim=%.5f, val_psnr=%.5f\n' % (step + 1, config.n_iters, image_loss, ssim, psnr))
                    # save weights with max ssim of validation dataset
                    if ssim > metric_ssim:
                        metric_ssim = ssim
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter': step,
                            'ssim': metric_ssim
                        }, checkpoint_dir)
                        print('save weights of step %d' % (step + 1))
                        log_file.write('save weights of step %d\n' % (step + 1))

            scheduler.step()

    if config.is_eval:
        # evaluation after training
        model.eval()
        with torch.no_grad():
            # load best model
            checkpoint = torch.load(checkpoint_dir, map_location=device)
            model.load_state_dict(checkpoint['model'])
            # evaluation after training
            for scale in config.eval_scale:
                eval_path = os.path.join(config.eval_path, 'scale={}'.format(scale))
                if not os.path.exists(eval_path):
                    os.makedirs(eval_path)
                # evaluation dataset
                eval_loader = data.get_dataloader(config, evaluation=True, eval_scale=scale)
                # ssim, psnr
                eval_ssim, eval_psnr = 0., 0.
                for i, batch in enumerate(eval_loader):
                    inputs, targets = batch

                    image = targets['image'].to(device)
                    # output generated data
                    pred_image = model(inputs['coords'].to(device),
                                       inputs['undersampled_image'].to(device),
                                       inputs['down_scale'].to(device)).permute(0, 3, 1, 2)
                    # evaluation metrics
                    eval_ssim += utils.ssim(pred_image, image).item()
                    eval_psnr += utils.psnr(pred_image, image).item()
                    # save output
                    pred_image = pred_image[0, 0].cpu().numpy()
                    np.savez(os.path.join(eval_path, targets['filename'][0]), pred_image=pred_image)
                # record the mean metrics
                num = len(eval_loader)

                print('\nEvaluation: downscale=%d ssim=%.4f and psnr=%.4f' % (scale, eval_ssim / num, eval_psnr / num))
                log_file.write('\nEvaluation: downscale=%d ssim=%.4f and psnr=%.4f\n' % (scale, eval_ssim / num, eval_psnr / num))

    log_file.close()
