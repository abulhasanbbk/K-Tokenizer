import os
import json
class Preporcess:
    def __init__(self,source, target, is_test_set=False, labeler=1, data_source="MIMIC-CXR"):
        self.source=source
        self.target=target
        self.data=self.load_json()
        self.text_key="text"
        self.entities_key="entities"
        self.labeller_key="labeler_"+str(labeler)
        #self.data_dict={key:None for key in self.data}
        self.data_dict={}
        self.is_test_set=is_test_set
        self.data_source=data_source
        self.parse()
        self.save_data_dict()
       
        #print(self.data_dict)
    def load_json(self):
        with open(self.source) as f:
            data=json.load(f)
        return data
    def parse(self):
        for key in self.data:
            text=self.data[key][self.text_key].split()
            if not self.is_test_set:
                entities=self.data[key][self.entities_key]
                labels=["O" for i in range(len(text))]
                for entity_id, entity_data in entities.items():
                    entity_label = entity_data['label']
                    start_ix = entity_data['start_ix']
                    end_ix = entity_data['end_ix']
                    labels[start_ix]=entity_label.replace("-","")
                    labels[end_ix]=entity_label.replace("-","")
                line=list(zip(text,labels))
                self.data_dict[key]=line
            else:
                src=self.data[key]["data_source"]
                if src==self.data_source:
                    entities=self.data[key][self.labeller_key][self.entities_key]
                    labels=["O" for i in range(len(text))]
                    for entity_id, entity_data in entities.items():
                        entity_label = entity_data['label']
                        start_ix = entity_data['start_ix']
                        end_ix = entity_data['end_ix']
                        labels[start_ix]=entity_label.replace("-","")
                        labels[end_ix]=entity_label.replace("-","")
                    line=list(zip(text,labels))
                    self.data_dict[key]=line
            
            
    def save_data_dict(self):
        placeholder= "___" #remove place holders from data
        max_length=20
        with open(target, 'w') as file:
            for key, values in self.data_dict.items():
                for word, label in values:
                    if word != placeholder and len(word)<=max_length:
                        file.write(f"{key} {word} {label}\n")
                file.write("\n")

        print(f"Data dictionary has been written to '{target}' successfully.")
if __name__ == "__main__":
    source="/working/abul/RadGraph/physionet.org/files/radgraph/1.0.0/train.json"
    target="data/train.txt"
    pr=Preporcess(source, target)
    source="/working/abul/RadGraph/physionet.org/files/radgraph/1.0.0/dev.json"
    target="data/dev.txt"
    pr=Preporcess(source, target)
    source="/working/abul/RadGraph/physionet.org/files/radgraph/1.0.0/test.json"
    target="data/test_MIMIC_CXR_labeller1.txt"
    pr=Preporcess(source, target, True, 1, "MIMIC-CXR")
    source="/working/abul/RadGraph/physionet.org/files/radgraph/1.0.0/test.json"
    target="data/test_MIMIC_CXR_labeller2.txt"
    pr=Preporcess(source, target, True, 2, "MIMIC-CXR")
    source="/working/abul/RadGraph/physionet.org/files/radgraph/1.0.0/test.json"
    target="data/test_CheXpert_labeller1.txt"
    pr=Preporcess(source, target, True, 1, "CheXpert")
    source="/working/abul/RadGraph/physionet.org/files/radgraph/1.0.0/test.json"
    target="data/test_CheXpert_labeller2.txt"
    pr=Preporcess(source, target, True, 2, "CheXpert")


