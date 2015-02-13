-- let's make a table of all of the movie reviews
-- to help us visualize the data we are working with
create table movie_ratings (
phrase_id integer,
sentence_id integer,
phrase varchar(500),
sentiment integer
);
-- note that we created a copy of the training data without the header
-- 'train2.tsv'
copy movie_ratings from '/Users/mshadish/kaggle/train2.tsv';

-- run some basic analysis, get a visual of the data
select count(1) from movie_ratings;

select phrase, count(1) from movie_ratings
group by phrase
--having count(1) = 1
order by count(1) desc;

select count(distinct sentence_id) from movie_ratings;

select sentiment, count(1) from movie_ratings group by sentiment order by sentiment;