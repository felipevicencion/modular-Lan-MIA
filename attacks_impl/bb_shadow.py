import torch 

class attack_bb_shadow_attack(shadow_attack):

    def run_attack(attack_dataset, membership_true, attack_options):
        self.num_shadow_models = attack_options['num_shadow_models']
        num_classes = model(self.shadow_datasets[0][0][0]).shape[1]
        logits = torch.empty((0, num_classes))
        membership = torch.empty((0, 1))
        
        for ind_data in range(self.num_shadow_models):
    
            model = transformers.AutoModelForSequenceClassification.from_pretrained(attack_options['load_model_dir'][ind_data])
            model.eval()
            # tokenizer = transformers.RobertaTokenizerFast.from_pretrained(attack_options['model_name'], max_length = attack_options['max_sequence_length'])
    
            # fetch train and test data
            train_data, test_data = self.shadow_datasets[ind_data][0], self.shadow_datasets[ind_data][1]
            train_data, test_data = tokenize_data(attack_options['tokenizer'], train_data, test_data, attack_options['max_sequence_length'])
    
            with torch.no_grad():
                new_logits = model(train_data)
                logits = torch.cat((logits, new_logits), dim=0)
                new_logits = model(test_data)
                logits = torch.cat((logits, new_logits), dim=0)
            membership = torch.cat((torch.ones(len(train_data)).view(-1, 1), torch.zeros(len(test_data)).view(-1, 1)), dim=0)
    
            layer_sizes = [5, 10, 5, 1]
            model = train_binary_classifier(layer_sizes, logits, membership, num_epochs=10, batch_size=16, learning_rate=0.001)
    
            trainer = attack_options['trainer']
            predictions_output = trainer.predict(attack_dataset)
            predictions = predictions_output.predictions
    
            model.eval()
            with torch.no_grad():
                shadow_prediction = model(predictions)
                self.membership_pred = (shadow_prediction >= 0.5).float()
    