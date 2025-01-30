from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions
from typing import List, Optional
from .WIDEN import merge
from .ActivationMerging import activation_merge


class Merger:
    def __init__(self, merge_method:str , base_model: str, models_to_merge: str, save_path: str, weights: Optional[List[float]] = None, dtype: str = "bfloat16", **merger_kwargs):
        self.base_model = base_model
        self.models_to_merge = models_to_merge
        self.save_path = save_path
        self.method = merge_method
        self.dtype = dtype
        self.merger_kwargs = merger_kwargs
        
        if weights is None:
            self.weights = [1.0 for _ in range(len(models_to_merge))]
        else:
            self.weights = weights
        
        self.models = []
        
        for i in range(len(models_to_merge)):
            self.models.append({
                'model': models_to_merge[i],
                'parameters': {'weight': self.weights[i]}
            })
        
        self.conf = {}
        self.conf['merge_method'] = merge_method
        if merge_method != 'linear':
            self.conf['base_model'] = base_model
        
        self.conf['models'] = self.models
        self.conf['dtype'] = dtype
        self.conf['chat_template'] = 'auto'
        self.conf_dict = self.conf
        self.conf = MergeConfiguration.model_validate(self.conf)

    def __call__(self, gpu=True):
        
        ## Pretty Pring the Configuration Dictionary
        print('|' +"-"*78+"|")
        print("|"+'-' * 31+" Merging Setup "+'-' * 32+"|")
        for key, value in self.conf_dict.items():
            if key == 'models':
                print('|' +"="*78+"|")
                p_string = f"{key}:"
                print('|'+p_string.center(78)+'|')
                for i, model in enumerate(value):
                    p_string = f"Model: {model['model']}"
                    print('|'+p_string.center(78)+'|')
                    
                    p_string = f"Parameters: {model['parameters']}"
                    print('|'+p_string.center(78)+'|')
                    
                    if i != len(value) - 1:
                        print('|' +"-"*78+"|")
                print('|' +"="*78+"|")
            else:
                print('|' +"-"*78+"|")
                p_string = f"{key}: {value}"
                print('|'+p_string.center(78)+'|')
        print('|' +"-"*78+"|")
        
        if self.method == 'WIDEN':
            merged_model, merged_tokenizer = merge(self.models_to_merge, self.base_model, self.method, **self.merger_kwargs)
            
            merged_model.save_pretrained(self.save_path)
            merged_tokenizer.save_pretrained(self.save_path)
            
            return
            
        elif self.method == 'AWR':
            merged_model, merged_tokenizer = activation_merge(self.models_to_merge, self.base_model, **self.merger_kwargs)
            
            merged_model.save_pretrained(self.save_path)
            merged_tokenizer.save_pretrained(self.save_path)
            
            return
            
        options = MergeOptions()
        if gpu:
            options.cuda = True
        else:
            options.cuda = False
            
        out_path = self.save_path
        run_merge(self.conf, out_path, options)
        