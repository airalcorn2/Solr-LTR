# **<p style="text-align: center;">From Zero to Learning to Rank in Apache Solr</p>**

This tutorial describes how to implement a modern [learning to rank](https://en.wikipedia.org/wiki/Learning_to_rank) (LTR) system in [Apache Solr](http://lucene.apache.org/solr/). The intended audience is people who have zero Solr experience, but who are comfortable with machine learning and information retrieval concepts. I was one of those people only a couple of months ago, and I found it extremely challenging to get up and running with the Solr materials available on the internet. This is my attempt at writing the tutorial I wish I had discovered.

# **<p style="text-align: center;">Table of Contents</p>**

* [**Setting Up Solr**](#Setting-Up-Solr)
* [**Solr Basics**](#Solr-Basics)
* [**Defining Features**](#Defining-Features)
* [**Learning to Rank**](#Learning-to-Rank)
* [**RankNet**](#RankNet)

# <a name="Setting-Up-Solr"></a>**<p style="text-align: center;">Setting Up Solr</p>**

Firing up a vanilla Solr instance on Fedora is actually pretty straightforward. First, download the Solr source tarball (so, one containing "src") from [here](http://lucene.apache.org/solr/mirrors-solr-latest-redir.html) and extract it to a reasonable location. Next, `cd` into the Solr directory:

```bash
cd /path/to/solr-<version>/solr
```

Building Solr requires [Apache Ant](http://ant.apache.org/) and [Apache Ivy](http://ant.apache.org/ivy/), so we'll have to install those:

```bash
sudo dnf install ant ivy
```

And now we'll build Solr.

```bash
ant ivy-bootstrap
ant server
```

You can confirm Solr is working by running:

```bash
bin/solr start
```

and making sure you see the Solr Admin interface at http://localhost:8983/solr/. You can stop Solr (but don't stop it now) with:

```bash
bin/solr stop
```

# <a name="Solr-Basics"></a>**<p style="text-align: center;">Solr Basics</p>**

Solr is a search platform, so we only really need to know how to do two things to function: **_(1)_** index data and **_(2)_** define a ranking model. Solr has a [REST](https://en.wikipedia.org/wiki/Representational_state_transfer)-like API, which means we'll be making changes with the [`curl`](https://curl.haxx.se/docs/manpage.html) command. To get going, let's first create a [core](https://lucene.apache.org/solr/guide/solr-cores-and-solr-xml.html) named `test`:

```bash
bin/solr create -c test
```

This seemingly simple command actually did a lot of stuff behind the scenes. Specifically, it defined a [schema](https://lucene.apache.org/solr/guide/documents-fields-and-schema-design.html#documents-fields-and-schema-design), which tells Solr how documents should be processed (think [tokenization](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html), [stemming](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html), etc.) and searched (e.g., using the [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) [vector space model](https://en.wikipedia.org/wiki/Vector_space_model)), and it set up a [configuration file](https://lucene.apache.org/solr/guide/configuring-solrconfig-xml.html), which specifies what libraries and handlers Solr will use. A core can be deleted with:

```bash
bin/solr delete -c test
```

OK, let's add some documents. First download [this XML file of tweets](https://github.com/treygrainger/solr-in-action/blob/master/example-docs/ch6/tweets.xml) provided on the [Solr in Action](http://solrinaction.com) GitHub. Take a look inside the XML file. Notice how it's using an `<add>` tag to tell Solr to add several documents (denoted with `<doc>` tags) to the index. To actually index the tweets, we run:

```bash
bin/post -c test /path/to/tweets.xml
```

Now, if we go to http://localhost:8983/solr/ (you might have to refresh) and click on the "Core Selector" dropdown on the left hand side, we can select the `test` core. If we then click on the "Query" tab, the query interface will appear. If we click on the blue "Execute Query" button at the bottom, a JSON document containing information regarding the tweets we just indexed will be displayed. Congratulations, you just ran your first successful query! Specifically, you used the [`/select` RequestHandler](https://lucene.apache.org/solr/guide/requesthandlers-and-searchcomponents-in-solrconfig.html) to execute the query `*:*`. The `*:*` is a special syntax that tells Solr to [return everything](https://stackoverflow.com/questions/8800380/solr-vs-query-performance). The [Solr query syntax](https://lucene.apache.org/solr/guide/query-syntax-and-parsing.html) is not very intuitive, in my opinion, so it's something you'll just have to get used to.

# <a name="Defining-Features"></a>**<p style="text-align: center;">Defining Features</p>**

OK, now that we have a basic Solr instance up and running, let's define some features for our LTR system. Like all machine learning problems, effective feature engineering is critical to success. Standard features in modern LTR models include using multiple similarity measures (e.g., [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of tf-idf vectors or [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)) to compare multiple text fields (e.g., body, title), in addition to other text characteristics (e.g., length) and document characteristics (e.g., age, PageRank). A good starting point is [this list of features](https://www.microsoft.com/en-us/research/project/mslr/) put together by Microsoft Research for an academic data set. A list of some other commonly used features can be found on slide 32 of [these lecture notes](https://people.cs.umass.edu/~jpjiang/cs646/16_learning_to_rank.pdf).

To start off, we're going to modify `/path/to/solr-<version>/solr/server/solr/test/conf/managed-schema` so that it includes the text fields that we'll need for our model. First, we'll change the `text` field so that it is of the `text_general` type (which is already defined inside `managed-schema`). The `text_general` type will allow us to calculate BM25 similarities. Because the `text` field already exists (it was automatically created when we indexed the tweets), we need to use the `replace-field` command like so:

```bash
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "replace-field" : {
     "name":"text",
     "type":"text_general",
     "indexed":"true",
     "stored":"true",
     "multiValued":"true"}
}' http://localhost:8983/solr/test/schema
```

I encourage you to take a look inside `managed-schema` following each change so that you can get a sense for what's happening. Next, we're going to specify a `text_tfidf` type, which will allow us to calculate tf-idf cosine similarities:

```bash
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field-type" : {
     "name":"text_tfidf",
     "class":"solr.TextField",
     "positionIncrementGap":"100",
     "indexAnalyzer":{
        "tokenizer":{
           "class":"solr.StandardTokenizerFactory"},
        "filter":{
           "class":"solr.StopFilterFactory",
           "ignoreCase":"true",
           "words":"stopwords.txt"},
        "filter":{
           "class":"solr.LowerCaseFilterFactory"}},
     "queryAnalyzer":{
        "tokenizer":{
           "class":"solr.StandardTokenizerFactory"},
        "filter":{
           "class":"solr.StopFilterFactory",
           "ignoreCase":"true",
           "words":"stopwords.txt"},
        "filter":{
           "class":"solr.SynonymGraphFilterFactory",
           "ignoreCase":"true",
           "synonyms":"synonyms.txt"},
        "filter":{
           "class":"solr.LowerCaseFilterFactory"}},
     "similarity":{
           "class":"solr.ClassicSimilarityFactory"}}
}' http://localhost:8983/solr/test/schema
```

Let's now add a `text_tfidf` field that will be of the `text_tfidf` type we just defined:

```bash
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field" : {
     "name":"text_tfidf",
     "type":"text_tfidf",
     "indexed":"true",
     "stored":"false",
     "multiValued":"true"}
}' http://localhost:8983/solr/test/schema
```

Because the contents of the `text` field and the `text_tfidf` field are the same (we're just handling them differently), we will tell Solr to copy the contents from `text` to `text_tfidf`:

```bash
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-copy-field" : {
     "source":"text",
     "dest":"text_tfidf"}
}' http://localhost:8983/solr/test/schema
```

We're now ready to re-index our data:

```bash
bin/post -c test /path/to/tweets.xml
```

# <a name="Learning-to-Rank"></a>**<p style="text-align: center;">Learning to Rank</p>**

Now that our documents are properly indexed, let's build a LTR model. If you're new to LTR, I recommend checking out [this (long) paper](http://didawiki.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/ir/ir13/1_-_learning_to_rank.pdf) by Tie-Yan Liu and [this textbook](https://www.springer.com/gp/book/9783642142666) also by Liu. If you're familiar with machine learning, the ideas shouldn't be too difficult to grasp. I also recommend checking out the [Solr documentation on LTR](https://lucene.apache.org/solr/guide/learning-to-rank.html), which I'll be linking to throughout this section. Enabling LTR in Solr first requires [making some changes](https://lucene.apache.org/solr/guide/learning-to-rank.html#LearningToRank-Installation) to `/path/to/solr-<version>/solr/server/solr/test/conf/solrconfig.xml`. Copy and paste the below text anywhere between the `<config>` and `</config>` tags (at the top and bottom of the file, respectively).

```xml
<lib dir="${solr.install.dir:../../../..}/contrib/ltr/lib/" regex=".*\.jar" />
<lib dir="${solr.install.dir:../../../..}/dist/" regex="solr-ltr-\d.*\.jar" />

<queryParser name="ltr" class="org.apache.solr.ltr.search.LTRQParserPlugin"/>

<cache name="QUERY_DOC_FV"
       class="solr.search.LRUCache"
       size="4096"
       initialSize="2048"
       autowarmCount="4096"
       regenerator="solr.search.NoOpRegenerator" />

<transformer name="features" class="org.apache.solr.ltr.response.transform.LTRFeatureLoggerTransformerFactory">
  <str name="fvCacheName">QUERY_DOC_FV</str>
</transformer>
```

We're now ready to run Solr with LTR enabled. First, stop Solr:

```bash
bin/solr stop
```

and then restart it with the LTR plugin enabled:

```bash
bin/solr start -Dsolr.ltr.enabled=true
```

Next, we need to push the model features and the model specification to Solr. In Solr, [LTR features are defined using a JSON formatted file](https://lucene.apache.org/solr/guide/learning-to-rank.html#LearningToRank-Uploadingfeatures). For our model, we'll save the following features in `my_efi_features.json`:

```json
[
  {
    "store" : "my_efi_feature_store",
    "name" : "tfidf_sim_a",
    "class" : "org.apache.solr.ltr.feature.SolrFeature",
    "params" : { "q" : "{!dismax qf=text_tfidf}${text_a}" }
  },
  {
    "store" : "my_efi_feature_store",
    "name" : "tfidf_sim_b",
    "class" : "org.apache.solr.ltr.feature.SolrFeature",
    "params" : { "q" : "{!dismax qf=text_tfidf}${text_b}" }
  },
  {
    "store" : "my_efi_feature_store",
    "name" : "bm25_sim_a",
    "class" : "org.apache.solr.ltr.feature.SolrFeature",
    "params" : { "q" : "{!dismax qf=text}${text_a}" }
  },
  {
    "store" : "my_efi_feature_store",
    "name" : "bm25_sim_b",
    "class" : "org.apache.solr.ltr.feature.SolrFeature",
    "params" : { "q" : "{!dismax qf=text}${text_b}" }
  },
  {
    "store" : "my_efi_feature_store",
    "name" : "max_sim",
    "class" : "org.apache.solr.ltr.feature.SolrFeature",
    "params" : { "q" : "{!dismax qf='text text_tfidf'}${text}" }
  },
  {
    "store" : "my_efi_feature_store",
    "name" : "original_score",
    "class" : "org.apache.solr.ltr.feature.OriginalScoreFeature",
    "params" : {}
  }
]
```

`store` tells Solr where to store the feature, `name` is the name of the feature, `class` specifies which [Java class will handle the feature](https://lucene.apache.org/solr/guide/learning-to-rank.html#LearningToRank-Featureengineering), and `params` provides additional information about the feature required by its Java class. In the case of a [`SolrFeature`](http://lucene.apache.org/solr/6_6_1/solr-ltr/org/apache/solr/ltr/feature/SolrFeature.html), you need to provide the query. `{!dismax qf=text_tfidf}${text_a}` tells Solr to search the `text_tfidf` field with the contents of `text_a` using the [`DisMaxQParser`](https://lucene.apache.org/solr/guide/the-dismax-query-parser.html). The reason we're using the DisMax parser instead of the seemingly more obvious [`FieldQParser`](https://lucene.apache.org/solr/guide/other-parsers.html#OtherParsers-FieldQueryParser) (e.g., `{!field f=text_tfidf}${text_a}`) is because the `FieldQParser` automatically converts multi-term queries to "phrases" (i.e., it converts something like "the cat in the hat" into, effectively, "the_cat_in_the_hat", rather than "the", "cat", "in", "the", "hat"). This `FieldQParser` behavior (which seems like a rather strange default to me) ended up [giving me quite a headache](https://issues.apache.org/jira/browse/SOLR-11386), but I eventually found a solution with `DisMaxQParser`.

`{!dismax qf='text text_tfidf'}${text}` tells Solr to search both the `text` and `text_tfidf` fields with the contents of `text` and then take the max of those two scores. While this feature doesn't really make sense in this context because we're already using similarities from both fields as features, it demonstrates how such a feature could be implemented. For example, imagine that the documents in your corpus are linked to, at most, five other sources of text data. It might make sense to incorporate that information during a search, and taking the max over multiple similarity scores is one way of doing that.

Finally, [`OriginalScoreFeature`](http://lucene.apache.org/solr/6_6_1/solr-ltr/org/apache/solr/ltr/feature/OriginalScoreFeature.html) "returns the original score that the document had before performing the reranking". This feature is necessary for returning the results in their original ranking when extracting features (**note**: `OriginalScoreFeature` [is broken](https://issues.apache.org/jira/browse/SOLR-11164) on Solr versions prior to 7.1).

To push the features to Solr, we run the following command:

```bash
curl -XPUT 'http://localhost:8983/solr/test/schema/feature-store' --data-binary "@/path/to/my_efi_features.json" -H 'Content-type:application/json'
```

If you ever want to upload new features, you have to first delete the old features with:

```bash
curl -XDELETE 'http://localhost:8983/solr/test/schema/feature-store/my_efi_feature_store'
```

Next, we'll save the following model specification in `my_efi_model.json`:

```json
{
  "store" : "my_efi_feature_store",
  "name" : "my_efi_model",
  "class" : "org.apache.solr.ltr.model.LinearModel",
  "features" : [
    { "name" : "tfidf_sim_a" },
    { "name" : "tfidf_sim_b" },
    { "name" : "bm25_sim_a" },
    { "name" : "bm25_sim_b" },
    { "name" : "max_sim" },
    { "name" : "original_score" }
  ],
  "params" : {
    "weights" : {
      "tfidf_sim_a" : 0.0,
      "tfidf_sim_b" : 0.0,
      "bm25_sim_a" : 0.0,
      "bm25_sim_b" : 0.0,
      "max_sim" : 0.0,
      "original_score" : 1.0
    }
  }
}
```

`store` specifies [where the features the model is using are stored](https://lucene.apache.org/solr/guide/learning-to-rank.html#LearningToRank-Lifecycle), `name` is the name of the model, `class` specifies which Java class will implement the model, `features` is a list of the model features, and `params` provides additional information required by the model's Java class. To start off with, we'll use the [`LinearModel`](https://lucene.apache.org/solr/6_6_1/solr-ltr/org/apache/solr/ltr/model/LinearModel.html), which simply takes a weighted sum of the feature values to generate a score. Here, we assign a weight of 0.0 to each feature except `original_score`, which is assigned a weight of 1.0. This weighting scheme will ensure the results are returned in their original order. To find  better weights, we'll need to extract training data from Solr. I'll go over this topic in more depth in the [RankNet section](#RankNet).

We can push the model to Solr with:

```bash
curl -XPUT 'http://localhost:8983/solr/test/schema/model-store' --data-binary "@/path/to/my_efi_model.json" -H 'Content-type:application/json'
```

And now we're ready to run our first LTR query:

<a href="http://localhost:8983/solr/test/query?q=historic north&df=text&rq={!ltr model=my_efi_model efi.text_a=historic efi.text_b=north efi.text='historic north'}&fl=id,score,[features]">`http://localhost:8983/solr/test/query?q=historic north&df=text&rq={!ltr model=my_efi_model efi.text_a=historic efi.text_b=north efi.text='historic north'}&fl=id,score,[features]`</a>

You should see something like:

```json
{
  "responseHeader":{
    "status":0,
    "QTime":1,
    "params":{
      "q":"historic north",
      "df":"text",
      "fl":"id,score,[features]",
      "rq":"{!ltr model=my_efi_model efi.text_a=historic efi.text_b=north efi.text='historic north'}"}},
  "response":{"numFound":1,"start":0,"maxScore":1.8617721,"docs":[
      {
        "id":"1",
        "score":1.8617721,
        "[features]":"tfidf_sim_a=0.35304558,tfidf_sim_b=0.0,bm25_sim_a=0.93088603,bm25_sim_b=0.93088603,max_sim=1.8617721,original_score=1.8617721"}]
  }}
```

Referring back to the request, `q=historic north` is the query used to fetch the initial results (using BM25 in this case), which are then re-ranked with the LTR model. `df=text` specifies the default field for Solr to search. `rq` is where all of the LTR parameters are provided. `efi` stands for "[external feature information](https://lucene.apache.org/solr/guide/learning-to-rank.html#LearningToRank-ExternalFeatureInformation)", which allows you to specify additional inputs at query time. In this case, we're populating the `text_a` argument with the term `historic`, the `text_b` argument with the term `north`, and the `text` argument with the multi-term query `'historic north'` (note, this is not being treated as a "phrase"). `fl=id,score,[features]` tells Solr to include the `id`, `score`, and model features in the results. You can verify that the feature values are correct by performing the associated search in the "Query" interface of the Solr Admin UI. For example, typing `text_tfidf:historic` in the `q` text box and typing `score` in the `fl` text box and then clicking the "Execute Query" button should return a value of 0.35304558.

# <a name="RankNet"></a>**<p style="text-align: center;">RankNet</p>**

For LTR systems, linear models are generally trained using what's called a "[pointwise](https://en.wikipedia.org/wiki/Learning_to_rank#Pointwise_approach)" approach, which is where documents are considered individually (i.e., the model asks, "Is this document relevant to the query or not?"); however, pointwise approaches are generally [not well-suited](https://www.quora.com/What-are-the-differences-between-pointwise-pairwise-and-listwise-approaches-to-Learning-to-Rank) for LTR problems. [RankNet](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf) is a neural network that uses a "[pairwise](https://en.wikipedia.org/wiki/Learning_to_rank#Pairwise_approach)" approach, which is where documents with a known relative preference are considered in pairs (i.e., the model asks, "Is document A more relevant than document B for the query or not?"). RankNet is available in Solr [as of version 7.3](https://github.com/apache/lucene-solr/commit/c5938f79e540f81b6d61560d324b150a5efd7011) (you can verify your version of Solr includes RankNet by inspecting `/path/to/solr-<version>/solr/dist/solr-ltr-{version}-SNAPSHOT.jar` and looking for `NeuralNetworkModel.class` under `/org/apache/solr/ltr/model/`). [I've also implemented RankNet in Keras](https://github.com/airalcorn2/RankNet) for model training. It's worth noting that [LamdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LambdaMART_Final.pdf) might be more appropriate for your particular search application. However, RankNet can be trained quickly on a GPU using Keras, which makes it a good solution for search problems where only one document is relevant to any given query. For a nice (technical) overview of RankNet, LambdaRank, and LambdaMART, see [this paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) by Chris Burges from (at the time) Microsoft Research.

Unfortunately, the [suggested method](https://lucene.apache.org/solr/guide/learning-to-rank.html#extracting-features) of feature extraction in Solr [is painfully slow](http://mail-archives.apache.org/mod_mbox/lucene-solr-user/201710.mbox/%3CCAMVOmhGUHZ-HU=g6o4+oyxob=bcWxQAg2BvW0Yj_sj=1bP0RRQ@mail.gmail.com%3E) ([other Solr users seem to agree it could be faster](http://mail-archives.apache.org/mod_mbox/lucene-solr-user/201710.mbox/%3C1508833512139-0.post@n3.nabble.com%3E)). Even when making the requests in parallel, it took me almost three days to extract features for ~200,000 queries. I think a better approach might be to do something like [this](http://computergodzilla.blogspot.com/2015/01/calculate-cosine-similarity-using.html), where you index the queries and then calculate the similarities between the "documents" (which consist of the true documents and queries), but this is really something that should be baked into Solr. Anyway, here is some Python template code for extracting features from Solr using queries (**note**: this code cannot be run as is):

```python
import numpy as np
import requests
import simplejson

# Number of documents to be re-ranked.
RERANK = 50
with open("RERANK.int", "w") as f:
    f.write(str(RERANK))

# Build query URL.
q_id = row["id"]
text_a = row["text_a"].strip().lower()
text_b = row["text_b"].strip().lower()
text = " ".join([text_a, text_b])

url = "http://localhost:8983/solr/test/query"
url += "?q={0}&df=text&rq={{!ltr model=my_efi_model ".format(text)
url += "efi.text_a='{0}' efi.text_b='{1}' efi.text='{2}'}}".format(text_a, text_b, text)
url += "&fl=id,score,[features]&rows={1}".format(text, RERANK)

# Get response and check for errors.
response = requests.request("GET", url)
try:
    json = simplejson.loads(response.text)
except simplejson.JSONDecodeError:
    print(q_id)

if "error" in json:
    print(q_id)

# Extract the features.
results_features = []
results_targets = []
results_ranks = []
add_data = False

for (rank, document) in enumerate(json["response"]["docs"]):

    features = document["[features]"].split(",")
    feature_array = []
    for feature in features:
        feature_array.append(feature.split("=")[1])

    feature_array = np.array(feature_array, dtype = "float32")
    results_features.append(feature_array)

    doc_id = document["id"]
    # Check if document is relevant to query.
    if q_id in relevant.get(doc_id, {}):
        results_ranks.append(rank + 1)
        results_targets.append(1)
        add_data = True
    else:
        results_targets.append(0)

if add_data:
    np.save("{0}_X.npy".format(q_id), np.array(results_features))
    np.save("{0}_y.npy".format(q_id), np.array(results_targets))
    np.save("{0}_rank.npy".format(q_id), np.array(results_ranks))
```

We're now ready to train some models. To start off with, we'll pull in the data and evaluate the BM25 rankings on the entire data set.

```python
import glob
import numpy as np

rank_files = glob.glob("*_rank.npy")
suffix_len = len("_rank.npy")

RERANK = int(open("RERANK.int").read())

ranks = []
casenumbers = []
Xs = []
ys = []
for rank_file in rank_files:
    X = np.load(rank_file[:-suffix_len] + "_X.npy")
    casenumbers.append(rank_file[:suffix_len])
    if X.shape[0] != RERANK:
        print(rank_file[:-suffix_len])
        continue

    rank = np.load(rank_file)[0]
    ranks.append(rank)
    y = np.load(rank_file[:-suffix_len] + "_y.npy")
    Xs.append(X)
    ys.append(y)

ranks = np.array(ranks)
total_queries = len(ranks)
print("Total Queries: {0}".format(total_queries))
print("Top 1: {0}".format((ranks == 1).sum() / total_queries))
print("Top 3: {0}".format((ranks <= 3).sum() / total_queries))
print("Top 5: {0}".format((ranks <= 5).sum() / total_queries))
print("Top 10: {0}".format((ranks <= 10).sum() / total_queries))
```

Next, we'll build and evaluate a (pointwise) linear support vector machine.

```python
from scipy.stats import rankdata
from sklearn.svm import LinearSVC

X = np.concatenate(Xs, 0)
y = np.concatenate(ys)

train_per = 0.8
train_cutoff = int(train_per * len(ranks)) * RERANK
train_X = X[:train_cutoff]
train_y = y[:train_cutoff]
test_X = X[train_cutoff:]
test_y = y[train_cutoff:]

model = LinearSVC()
model.fit(train_X, train_y)
preds = model._predict_proba_lr(test_X)

n_test = int(len(test_y) / RERANK)
new_ranks = []
for i in range(n_test):
    start = i * RERANK
    end = start + RERANK
    scores = preds[start:end, 1]
    score_ranks = rankdata(-scores)
    old_rank = np.argmax(test_y[start:end])
    new_rank = score_ranks[old_rank]
    new_ranks.append(new_rank)

new_ranks = np.array(new_ranks)
print("Total Queries: {0}".format(n_test))
print("Top 1: {0}".format((new_ranks == 1).sum() / n_test))
print("Top 3: {0}".format((new_ranks <= 3).sum() / n_test))
print("Top 5: {0}".format((new_ranks <= 5).sum() / n_test))
print("Top 10: {0}".format((new_ranks <= 10).sum() / n_test))
```

Now we can try out RankNet. First we'll assemble the training data so that each row consists of a relevant document vector concatenated with an irrelevant document vector (for a given query). Because we returned 50 rows in the feature extraction phase, each query will have 49 document pairs in the data set.

```python
Xs = []
for rank_file in rank_files:
    X = np.load(rank_file[:-suffix_len] + "_X.npy")
    if X.shape[0] != RERANK:
        print(rank_file[:-suffix_len])
        continue

    rank = np.load(rank_file)[0]
    pos_example = X[rank - 1]
    for (i, neg_example) in enumerate(X):
        if i == rank - 1:
            continue
        Xs.append(np.concatenate((pos_example, neg_example)))

X = np.stack(Xs)
dim = int(X.shape[1] / 2)

train_per = 0.8
train_cutoff = int(train_per * len(ranks)) * (RERANK - 1)

train_X = X[:train_cutoff]
test_X = X[train_cutoff:]
```

Here, we build the model in Keras.

```python
from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Add, Dense, Input, Lambda
from keras.models import Model

y = np.ones((train_X.shape[0], 1))

INPUT_DIM = 5
h_1_dim = 64
h_2_dim = h_1_dim // 2
h_3_dim = h_2_dim // 2

# Model.
h_1 = Dense(h_1_dim, activation = "relu")
h_2 = Dense(h_2_dim, activation = "relu")
h_3 = Dense(h_3_dim, activation = "relu")
s = Dense(1)

# Relevant document score.
rel_doc = Input(shape = (INPUT_DIM, ), dtype = "float32")
h_1_rel = h_1(rel_doc)
h_2_rel = h_2(h_1_rel)
h_3_rel = h_3(h_2_rel)
rel_score = s(h_3_rel)

# Irrelevant document score.
irr_doc = Input(shape = (INPUT_DIM, ), dtype = "float32")
h_1_irr = h_1(irr_doc)
h_2_irr = h_2(h_1_irr)
h_3_irr = h_3(h_2_irr)
irr_score = s(h_3_irr)

# Subtract scores.
negated_irr_score = Lambda(lambda x: -1 * x, output_shape = (1, ))(irr_score)
diff = Add()([rel_score, negated_irr_score])

# Pass difference through sigmoid function.
prob = Activation("sigmoid")(diff)

# Build model.
model = Model(inputs = [rel_doc, irr_doc], outputs = prob)
model.compile(optimizer = "adagrad", loss = "binary_crossentropy")
```

And now to train and test the model.


```python
NUM_EPOCHS = 30
BATCH_SIZE = 32
checkpointer = ModelCheckpoint(filepath = "valid_params.h5", verbose = 1, save_best_only = True)
history = model.fit([train_X[:, :dim], train_X[:, dim:]], y,
                     epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, validation_split = 0.05,
                     callbacks = [checkpointer], verbose = 2)

model.load_weights("valid_params.h5")
get_score = backend.function([rel_doc], [rel_score])
n_test = int(test_X.shape[0] / (RERANK - 1))
new_ranks = []
for i in range(n_test):
    start = i * (RERANK - 1)
    end = start + (RERANK - 1)
    pos_score = get_score([test_X[start, :dim].reshape(1, dim)])[0]
    neg_scores = get_score([test_X[start:end, dim:]])[0]

    scores = np.concatenate((pos_score, neg_scores))
    score_ranks = rankdata(-scores)
    new_rank = score_ranks[0]
    new_ranks.append(new_rank)

new_ranks = np.array(new_ranks)
print("Total Queries: {0}".format(n_test))
print("Top 1: {0}".format((new_ranks == 1).sum() / n_test))
print("Top 3: {0}".format((new_ranks <= 3).sum() / n_test))
print("Top 5: {0}".format((new_ranks <= 5).sum() / n_test))
print("Top 10: {0}".format((new_ranks <= 10).sum() / n_test))

# Compare to BM25.
old_ranks = ranks[-n_test:]
print("Total Queries: {0}".format(n_test))
print("Top 1: {0}".format((old_ranks == 1).sum() / n_test))
print("Top 3: {0}".format((old_ranks <= 3).sum() / n_test))
print("Top 5: {0}".format((old_ranks <= 5).sum() / n_test))
print("Top 10: {0}".format((old_ranks <= 10).sum() / n_test))
```

If the model's results are satisfactory, we can save the parameters to a JSON file to be pushed to Solr:

```python
import json

weights = model.get_weights()
solr_model = {"store" : "my_efi_feature_store",
              "name" : "my_ranknet_model",
              "class" : "org.apache.solr.ltr.model.NeuralNetworkModel",
              "features" : [
                { "name" : "tfidf_sim_a" },
                { "name" : "tfidf_sim_b" },
                { "name" : "bm25_sim_a" },
                { "name" : "bm25_sim_b" },
                { "name" : "max_sim" }
              ],
              "params": {}}
layers = []
layers.append({"matrix": weights[0].T.tolist(),
               "bias": weights[1].tolist(),
               "activation": "relu"})
layers.append({"matrix": weights[2].T.tolist(),
               "bias": weights[3].tolist(),
               "activation": "relu"})
layers.append({"matrix": weights[4].T.tolist(),
              "bias": weights[5].tolist(),
              "activation": "relu"})
layers.append({"matrix": weights[6].T.tolist(),
              "bias": weights[7].tolist(),
              "activation": "identity"})
solr_model["params"]["layers"] = layers

with open("my_ranknet_model.json", "w") as out:
    json.dump(solr_model, out, indent = 4)
```

and it's pushed the same as before:

```bash
curl -XPUT 'http://localhost:8983/solr/test/schema/model-store' --data-binary "@/path/to/my_ranknet_model.json" -H 'Content-type:application/json'
```

We can also perform an LTR query like before, except this time we'll use `ltr_model=my_ranknet_model`.

<a href="http://localhost:8983/solr/test/query?q=historic north&df=text&rq={!ltr model=my_ranknet_model efi.text_a=historic efi.text_b=north efi.text='historic north'}&fl=id,score,[features]">`http://localhost:8983/solr/test/query?q=historic north&df=text&rq={!ltr model=my_ranknet_model efi.text_a=historic efi.text_b=north efi.text='historic north'}&fl=id,score,[features]`</a>

And there you have it &mdash; a modern learning to rank setup in Apache Solr.
