from agents import research_agent, writer_agent, editor_agent, investigar_y_escribir, planner_agent   # importa la función recién armada

# =============================================Prueba para investigador ===========================================================================

##resultado = research_agent("¿Qué es la malaria?")   # una pregunta que debería usar Wikipedia
#resultado = research_agent("Busca papers científicos recientes sobre biosensores para diagnóstico de malaria.") # una que deberia usar PubMed
##resultado = research_agent("¿Cuánto es 7 más 5?")
#print(resultado)
##print(resultado)
##print(resultado)

# =============================================Prueba para escritor ===========================================================================
#tema = "biosensores para diagnóstico de malaria"
#hallazgos = research_agent(f"Busca papers científicos sobre {tema}.")   # primero investiga
#resumen = writer_agent(tema, hallazgos)                                  # luego escribe con lo que encontró
#
#print(resumen)

# =============================================Prueba para revisor ==========================================================================
#tema = "biosensores para diagnóstico de malaria"
#hallazgos = research_agent(f"Busca papers científicos sobre {tema}.")
#resumen = writer_agent(tema, hallazgos)
#revision = editor_agent(tema, resumen)
#
#print("RESUMEN:")
#print(resumen)
#print()
#print("REVISIÓN DEL EDITOR:")
#print(revision)
# ==============================================Prueba para investigar y escribir ==========================================================================
#from agents import investigar_y_escribir
#
#resultado_final = investigar_y_escribir("biosensores para diagnóstico de malaria")
#print(resultado_final)

# =============================================Prueba para Planificador 1.1 ==========================================================================
#from agents import planner_agent, investigar_y_escribir
#
#sub_temas = planner_agent("Quiero un panorama del estado del arte en biosensores ópticos para diagnóstico de malaria.")
#
#for sub_tema in sub_temas:
#    print(f"\n=== {sub_tema} ===")
#    resultado = investigar_y_escribir(sub_tema)
#    print(resultado)

# =============================================Prueba para Planificador1.2 ==========================================================================

#from agents import investigar_y_escribir
#
#resultado = investigar_y_escribir("biosensores ópticos para diagnóstico de malaria")
#print(resultado)

# =============================================Prueba para Planificador1.3 ==========================================================================


#from agents import planner_agent
#
#sub_temas = planner_agent("Quiero un panorama del estado del arte en biosensores ópticos para diagnóstico de malaria.")
#print(sub_temas)

# =============================================Prueba para añadir archivo csv ==========================================================================
from agents import research_agent

resultado = research_agent("¿Qué es la malaria?")
print(resultado)