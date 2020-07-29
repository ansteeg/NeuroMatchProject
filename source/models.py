#%%
from hcp_dataset import *

task = 'gambling'
#networks = ['Ventral Mult']
conditions = ["win_event", "loss_event"]
#subjects = range(3) 

# Select trials
trials = []
for subject in subjects:
  subject_data = load_timeseries(subject, task, concat=False)
  for condition in conditions:
    evs = load_evs(subject, task, condition)
    #frames = condition_frames(evs, skip=0)
    selected = select_trials_hrf(subject_data, evs, hemo_dynamic_delay=6)
    selected = [np.reshape(x, (360, -1)) for x in selected]
    selected = np.hstack(selected)
    # Structure of selective output:
    # list of 2 runs
    # each run list of regions
    # each region list of timestamps
    # Reshaped into 360 regions * n timestamps array
    # Iterate over timestamps
    for i in range(selected.shape[1]):
      trials.append([subject, selected[:,i], condition])
trials = pd.DataFrame(trials, columns=['subject', 'regionValues', 'condition'])
#print(trials)

#%%
# Basic logistic regression
## Run model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import l1_min_c

X = np.array(trials['regionValues'])
X = np.array([np.array(xi) for xi in X])
# Add subject number as feature
#X = np.hstack([X, trials['subject'].reshape(-1,1)])
y = trials['condition'] == 'win_event'
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Create model
model = LogisticRegression(random_state=0, penalty='l1', solver='liblinear')
# Regularization parameter (seems not to affect much)
#model.set_params(C=l1_min_c(X_train, y_train, loss='log')*np.log(-3)) 
# Train the model using the training sets
model.fit(X_train, y_train)
# Make predictions using the testing set
acc = model.score(X_test, y_test)
print(f'Accuracy: {acc}')
# The coefficients
#print('Coefficients: \n', regr.coef_)


#%%
# Logistic regression with L1 param search cross validation
## Run model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import l1_min_c

X = np.array(trials['regionValues'])
X = np.array([np.array(xi) for xi in X])
# Add subject number as feature
#X = np.hstack([X, trials['subject'].reshape(-1,1)])
y = trials['condition'] == 'win_event'
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Create model
model = LogisticRegression(random_state=0, penalty='l1', solver='liblinear',
                                      tol=1e-6, max_iter=int(1e6),
                                      warm_start=True,
                                      intercept_scaling=10000.)
# Regularization parameter
cs = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 3, 16)
coefs_ = []
accs = []
# Iterate over smaller subset for parameter search to reduce computational time
X_train_subset = X_train[:5000]
Y_train_subset = y_train[:5000]
for c in cs:
    model.set_params(C=c)
    model.fit(X_train_subset, Y_train_subset)
    coefs_.append(model.coef_.ravel().copy())
    accs.append(model.score(X_test, y_test))


# Plot coefs
coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_, marker='o')
ymin, ymax = plt.ylim()
plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.title('Logistic Regression Path')
plt.axis('tight')
plt.show()

# Plot number of non-zero coefs and accuracies
coefs_ = np.count_nonzero(coefs_, axis=1)
fig, ax1 = plt.subplots()
ax1.set_xlabel('log(C)')
ax1.set_ylabel('Coefficients')
ax1.plot(np.log10(cs), coefs_, marker='o')


ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(np.log10(cs), accs, color=color)

plt.title('Number of non-zero coefficients / accuracy')
fig.tight_layout()
plt.show()

# Train the model using the training sets
#model.fit(X_train, y_train)
# Make predictions using the testing set
#acc = model.score(X_test, y_test)
#print(f'Accuracy: {acc}')
# The coefficients
#print('Coefficients: \n', regr.coef_)


# %%
# Neural Net

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X = np.array(trials['regionValues'])
X = np.array([np.array(xi) for xi in X])
# Add subject number as feature
#X = np.hstack([X, trials['subject'].reshape(-1,1)])
y = trials['condition'] == 'win_event'
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MLPClassifier(solver='lbfgs', alpha=1e2, hidden_layer_sizes=(10, 5, 2), random_state=1)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print(acc)

# %%
# Run logistic regression with only one parcel or all except one to see parcel contribution.
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X = np.array(trials['regionValues'])
X = np.array([np.array(xi) for xi in X])
y = trials['condition'] == 'win_event'
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Run full model
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
acc_total = model.score(X_test, y_test)
print(f'Full accuracy: {acc}')

region_scores = []
for region in region_info['name']:
  other_regions = region_info['name'][region_info['name'] != region]
  # Run partial model
  model = LogisticRegression(random_state=0)
  model.fit(X_train, y_train)
  acc = model.score(X_test, y_test)
  unique_contribution = acc_total - acc
  region_scores.append((region, unique_contribution))
#Save to file
region_scores = sorted(region_scores, key=lambda x: x[1])
with open('regionScores.csv','w') as out:
  csv_out=csv.writer(out)
  csv_out.writerow(['name','num'])
  for row in data:
    csv_out.writerow(row)