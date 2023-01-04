
import argparse
import logging
import re

#import pickle
#from sklearn.ensemble import RandomForestRegressor
#

import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions


class FormatInput(beam.DoFn):
    """Format the output"""
    def process(self, element):
        values = re.split(",", re.sub('\r\n', '', re.sub('"', '',element)))
        row = dict(zip(('time','temp','dwpt','rhum','prcp', 'snow','wdir','wspd','wpgt','pres','coco', 'el_price','consumption'), values))

        return [row]
        

class DataEnrichment(beam.DoFn):
    """Enrich data"""
    
    def process(self, element):
        from dateutil import parser
        from datetime import datetime
        import math

        element['time'] = parser.parse(element['time'])
        element['hr'] = element['time'].hour
        element['month'] = element['time'].month
        element['yday'] = element['time'].toordinal() - datetime(element['time'].year, 1, 1).toordinal() + 1 
        element['wday'] = element['time'].weekday() 
        element['day'] = element['time'].day

        element['hr_cos'] = math.cos(element['hr'] / 12 * math.pi )
        element['hr_sin'] = math.sin(element['hr'] / 12* math.pi ) 
        element['day_cos'] = math.cos(element['time'].day *2 * math.pi / (365.25/12))
        element['day_sin'] = math.cos(element['time'].day *2 * math.pi / (365.25/12))
        element['yday_cos'] = math.cos(element['yday'] * 2* math.pi / 365.25) 
        element['yday_sin'] = math.sin(element['yday'] * 2* math.pi / 365.25)


        return [element]



def download_blob(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    
    from google.cloud import storage
    
    destination_file_name = source_blob_name
    source_blob_name = "model/" + source_blob_name
    storage_client = storage.Client("eesti-energia-372414")
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)




class ConsumptionPrediction(beam.DoFn):
    """ Inicialise Model from pickle and use it to predict, with expm1 transformation"""
    def __init__(self):
        self.model = None
    
    def setup(self):
        import pickle
        from sklearn.ensemble import RandomForestRegressor
        logging.info("Starting up Setup, pray Odin")
        model_name = 'finalized_model_bucket.sav'
        download_blob(bucket_name= "eesti-energia-datalake", source_blob_name = model_name)
        self._model = pickle.load(open(model_name,'rb'))

    def process(self,element):
        import math

        variables = self._model.feature_names_in_

        list_values = []

        for x in variables:
            if x in [*element.keys()]:
                list_values.append(element[x])
            else:
                list_values.append(0)

        element['prediction'] = math.expm1(self._model.predict([list_values])[0])

        element_simple = dict()
        element_simple['time'] = element['time']
        element_simple['prediction'] = element['prediction']

        return [element_simple]



def get_cloud_pipeline_options():
    """Get apache beam pipeline options to run with Dataflow on the cloud
    """
    options = {
        'runner': 'DataflowRunner',
        'input': 'gs://eesti-energia-datalake/raw/test_small.csv',
        'job_name': 'bucket-to-sql-predict-small5968',
        'staging_location': "gs://eesti-energia-datalake/staging",
        'temp_location':  "gs://eesti-energia-datalake/temp",
        'project': "eesti-energia-372414",
        'region': 'europe-north1',
        'autoscaling_algorithm': 'THROUGHPUT_BASED',
        'max_num_workers': 2,
        'requirements_file': './requirements.txt'
#        'setup_file': './setup.py'
    }
    return beam.pipeline.PipelineOptions(flags=[], **options)



def printy(x):
    print(x)
    logging.info('predictions:{}'.format(x))


def run(argv = None):
    """
    Runs pipeline
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        dest = 'input',
        required = False,
        help = 'input file to read',
        default = 'gs://eesti-energia-datalake/raw/test_small.csv'
    )

    parser.add_argument(
        '--output',
        dest = 'output',
        required = False,
        help = 'Output BQ Table to write results to',
        default = 'eesti-energia-372414.train_enriched.enriched_test_all_on_dataflow_simple424'
    )

    parser.add_argument(
        '--runner',
        dest = 'runner',
        required= False,
        help = 'specificy runner',
        default= "DataflowRunner"
    )

    parser.add_argument(
        '--project',
        dest = 'project',
        required= False,
        help = 'project id',
        default= "eesti-energia-372414"
    )

    parser.add_argument(
        '--job_name',
        dest = 'job_name',
        required= False,
        help = 'Job name for job in dataFlow',
        default= "raw-to-bq"
    )

    parser.add_argument(
        '--region',
        dest = 'region',
        required= False,
        help = 'name of region',
        default= "europe-north1"
    )
    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = get_cloud_pipeline_options()
    pipeline_options.view_as(SetupOptions).save_main_session = True

    p = beam.Pipeline(options = pipeline_options)
    schema_table = 'time:TIMESTAMP,prediction:FLOAT'
    (p
     | 'Read from a File' >> beam.io.ReadFromText(known_args.input,
                                                  skip_header_lines=1)
     | 'String To BigQuery Row' >> beam.ParDo(FormatInput())
     | 'Data Enrichment' >> beam.ParDo(DataEnrichment())
     | 'Predict using Sklearn' >> beam.ParDo(ConsumptionPrediction())
     | 'Write to BigQuery' >>
     beam.io.WriteToBigQuery(
        known_args.output,
        schema=schema_table,
        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE
     ))

    p.run().wait_until_finish()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()