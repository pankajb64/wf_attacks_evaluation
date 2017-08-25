# wf_attacks_evaluation

Evaluation of the different methodologies for Machine-Learning based Website Fingerprinting Attacks on the Tor Browser. Independent Study under Prof Amir Houmansadr at The Secure, Private Internet Group (SPIN), University of Massacusetts, Amherst.

# References -

* Juarez, Marc, et al. "A critical evaluation of website fingerprinting attacks." Proceedings of the 2014 ACM SIGSAC Conference on Computer and Communications Security. ACM, 2014. http://www1.icsi.berkeley.edu/~sadia/papers/ccs-webfp-final.pdf
* Wang, Tao, and Ian Goldberg. "Improved website fingerprinting on tor." Proceedings of the 12th ACM workshop on Workshop on privacy in the electronic society. ACM, 2013. http://www.cypherpunks.ca/~iang/pubs/webfingerprint-wpes.pdf

# See also
**webfp-crawler-phantomjs** - The crawler used to collect data for this study. https://github.com/pankajb64/webfp-crawler-phantomjs/

# Summary

As a part of this project, I looked at the existing methods for targeting Website Fingerprinting (WF) Attacks on the Tor network, with a focus on the model by [Wang and Goldberg](http://www.cypherpunks.ca/~iang/pubs/webfingerprint-wpes.pdf) which SVMs with a custom kernel based on Optimal String Alignment Distance (OSAD). I also chiefly looked at the work of [Juarez et. al.](http://www1.icsi.berkeley.edu/~sadia/papers/ccs-webfp-final.pdf) which highlights the assumptions made by these models and their performance in their absence. To verify this end, I attempted to collect a dataset of website crawls using the Alexa Top 100 URLs, initially using phantomjs, and later using the Tor Browser client itself. I then ran the SVM code provided by Wang and Goldberg which gave me an accuracy of ~86%. 

I then attempted to tackle the multi-tab assumption described by Juarez et al. (i.e. browsing more than one web pages at a time on the Tor browser). I collected the crawls for the same, and wrote the code for pre-processing the data, transforming it into input instances suitable for training. The code to crawl the website is adapted from Juarez et. al. While collecting the crawl it collects website traces using wireshark. I then replace each packet by its length and divide each length by 512, and finally replace each length by a sequence of 1 (incoming packets) and -1 (outgoing packets) (i.e. if the outgoing packet length is 3, its replaced by [1, 1, 1]). This is the same approach taken by Wang and Juarez. I tried two neural network based models for classification, one a simple 1-hidden layer network and other a 1D CNN network (a simplified, trimmed down version of VGG-16). 

For the experiment, I tried to create a simplified version of an open-world scenario, where the attacker has a set of k webpages that he wishes to monitor. The attacker trains the model to output a 1 if the website trace matches any of the webpages in the monitored set, else output a 0. I also tried to limit the website trace input to the initial l number of packets, since I think the initial part of a website might contain the most data to coarsely identify two websites. Also, the usual length of packet counts of websites can go higher than 20k (especially for multi-tab data) and training on the entire length of the trace would require collecting instances that are exponential in number, hence it made sense to limit the number of packet counts fed to the network. I trained the network with varying values of l from 500 to 10000 and also for varying number of epochs (from 1 to 20). Training the neural networks gave an accuracy of ~91% on the single tab data and ~81% on the multi-tab data. Both these results were highest when the packet counts were in the range from 6-8k.
