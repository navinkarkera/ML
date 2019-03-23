import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score

plt.style.use("ggplot")

cols = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]

df = pd.read_csv("corrected", sep=",", names=cols + ["label"], index_col=None)
df = df[df["service"] == "http"]
df = df.drop("service", axis=1)
cols.remove("service")

encs = dict()
data = df.copy()
for c in data.columns:
    if data[c].dtype == "object":
        encs[c] = LabelEncoder()
        data[c] = encs[c].fit_transform(data[c])

X_train, X_test, y_train, y_test = train_test_split(data[cols], data["label"], test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

clf = IsolationForest(behaviour="new", contamination="auto", max_samples=256, random_state=2018)
clf.fit(X_train)
X_train["predicted"] = clf.predict(X_train)
scores = clf.decision_function(X_val)
plt.figure(figsize=(12, 8))
plt.hist(scores, bins=50)

print("AUC: {:.1%}".format(roc_auc_score((-0.2 < scores), y_val == list(encs["label"].classes_).index("normal."))))
scores_test = clf.decision_function(X_test)
print("AUC: {:.1%}".format(roc_auc_score((-0.2 < scores_test), y_test == list(encs["label"].classes_).index("normal."))))
plt.show()
