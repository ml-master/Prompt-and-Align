def train_model(args, x_train, x_test, y_train, y_test, tokenizer, max_len, n_epochs, batch_size, datasetname, iter, use_tpl=True):
    model = BERTPrompt().to(device)
    optimizer = AdamW(model.parameters(), lr = 5e-5)
    train_loader = create_train_loader(x_train, y_train, tokenizer, max_len, batch_size)
    test_loader = create_eval_loader(x_test, y_test, tokenizer, max_len, batch_size)

    total_steps = len(train_loader) * n_epochs

    if len(x_train) == 16:
        total_steps = 5 * n_epochs
        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    for epoch in range(n_epochs):
        model.train()
        avg_loss = []
        avg_acc = []
        for Batch_data in tqdm(train_loader):

            input_ids = Batch_data["input_ids"].to(device)
            targets = Batch_data["labels"].to(device)
            target_logits = torch.nonzero(targets)[:,-1]

            out_labels = model(input_ids = input_ids, masked_position = Batch_data["masked_pos"][0].item())

            loss_func = nn.BCELoss()
            loss = loss_func(out_labels,targets)

            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
            _, pred = out_labels.max(dim = -1)
            correct = pred.eq(target_logits).sum().item()
            train_acc = correct / len(targets)
            avg_acc.append(train_acc)
        
        train_loss = np.mean(avg_loss)
        train_acc = np.mean(avg_acc)

        print("Iter {:03d} | Epoch {:05d} | Train Acc. {:.4f} | Train_Loss {:.4f}".format(iter, epoch, train_acc, train_loss))


    model.eval()
    y_pred = []
    y_test = []
    y_probs = []
    
    for Batch_data in tqdm(test_loader):
        with torch.no_grad():
            # ... [省略其他代码部分] ...
            test_out = model(input_ids=input_ids, masked_position=Batch_data["masked_pos"][0].item())
            _, val_pred = test_out.max(dim=-1)

            y_pred.append(val_pred)
            y_test.append(target_logits)
            y_probs.append(test_out)

    # 如果不使用TPL，直接使用模型输出的概率
    if not use_tpl:
        y_probs = torch.cat(y_probs, dim=0)
        _, y_pred_upd = y_probs.max(dim=1)
    else:
        # 使用TPL的原始代码
        y_probs = torch.cat(y_probs, dim=0)
        for i in range(y_probs.shape[0]):
            if y_probs[i][0] - y_probs[i][1] >= threshold:
                y_probs[i][0] = 1.
                y_probs[i][1] = 0.
            if y_probs[i][1] - y_probs[i][0] >= threshold:
                y_probs[i][0] = 0.
                y_probs[i][1] = 1.
        y_probs = torch.cat([train_conf, y_probs], dim = 0)                      

        
        correct = y_pred.eq(y_test).sum().item()
        test_acc = correct / len(y_test)

        print("Iter {:03d} | Epoch {:05d} | Prompting Test Acc. {:.4f}".format(iter, epoch, test_acc))

        if epoch == n_epochs - 1:

            print("propagation on news-news graph")

            y_probs_upd = torch.matmul(A_nn, y_probs)

            y_probs_upd = torch.matmul(A_nn, y_probs_upd)
            _, y_pred_upd = y_probs_upd.max(dim=1)
            y_pred_upd = y_pred_upd[n_samples:]
            
            real_index = y_test.data==0
            correct_real = y_pred_upd[real_index].eq(y_test[real_index]).sum().item()
            correct_rumor = y_pred_upd[~real_index].eq(y_test[~real_index]).sum().item()
            test_acc_real = correct_real / real_index.sum().item()
            test_acc_rumor = correct_rumor /( ~real_index).sum().item()

            acc = accuracy_score(y_test.detach().cpu().numpy(), y_pred_upd.detach().cpu().numpy())
            precision, recall, fscore, _ = score(y_test.detach().cpu().numpy(), y_pred_upd.detach().cpu().numpy(), average='macro')
            

    print("-----------------End of Iter {:03d}-----------------".format(iter))
    print(['Final Test Accuracy:{:.4f}'.format(acc),
        'Precision:{:.4f}'.format(precision),
        'Recall:{:.4f}'.format(recall),
        'F1:{:.4f}'.format(fscore)])
    print(f"Final: Test Acc real:{test_acc_real} | Test Acc rumor:{test_acc_rumor}")
    
    return acc, precision, recall, fscore,test_acc_real,test_acc_rumor