| Risk           | Example                                      | Impact      | Mitigation                                        |
| -------------- | -------------------------------------------- | ----------- | ------------------------------------------------- |
| Data leakage   | Future customer activity used for prediction | High        | Only use information available at prediction time |
| False positive | Contact a customer unnecessarily             | Medium      | Control threshold and intervention cost           |
| False negative | Miss a customer who later churns             | High        | Monitor recall                                    |
| Privacy        | Customer information exposed                 | High        | Limit access and remove unnecessary identifiers   |
| Small dataset  | Poor generalization                          | High        | Collect more representative data                  |
| Segment bias   | Performance differs by plan                  | Medium      | Evaluate metrics by segment                       |
| Model drift    | Customer behavior changes                    | Medium/High | Monitor data and model performance                |
